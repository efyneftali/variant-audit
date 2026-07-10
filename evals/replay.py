"""Record-and-replay layer for the five external evidence tools (VA-39).

The eval grades the agent by calling the same five network tools the agent uses
(ClinVar, gnomAD, Ensembl VEP, UCSC, AlphaMissense). Hitting those live on every
eval run is the exact thing that made VA-20's numbers untrustworthy: NCBI and
gnomAD rate-limit under sustained load, rows error out and drop from the
denominator, and recall@k ends up computed over whichever rows happened to
survive (see evals/reports/ -- a "recall 1.0" run that actually scored 12 of 36
retrieval rows).

This layer freezes each tool's response per variant into evals/fixtures/ once,
then replays it from disk on every subsequent run. Two modes:

  replay (default): each tool call reads evals/fixtures/{tool}/{variant}.json.
      No network. A missing fixture is a hard error (FixtureMissing), never a
      silent gap -- a hole reintroduces exactly the loss this kills.
  record (explicit): call the real tools, sleep between calls to stay under the
      rate limits, retry failures rather than accept a gap, and write the
      responses to fixtures.

Keyed by (tool, variant): the tool's response is a deterministic function of the
variant, so that pair is the whole cache key. The two tools that also take a
`consequence=` hint (ucsc, alphamissense) get it ignored on replay -- it's a
round-trip optimization derived from the same variant's ensembl fixture, not an
independent input.

The agent's live path is untouched: nothing here modifies src/. install() only
rebinds the five names inside the already-imported `graph` module, and only in
the eval process that opts in. Production imports of graph/mcp_tools are
unaffected.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.variant_audit import graph  # noqa: E402
from src.variant_audit.mcp_tools.alphamissense import get_alphamissense_score  # noqa: E402
from src.variant_audit.mcp_tools.clinvar import get_clinvar_record  # noqa: E402
from src.variant_audit.mcp_tools.ensembl import get_gene_consequence  # noqa: E402
from src.variant_audit.mcp_tools.gnomad import get_allele_frequency  # noqa: E402
from src.variant_audit.mcp_tools.ucsc import get_genomic_context  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
DATASET = Path(__file__).parent / "golden_dataset.jsonl"

# tool key -> (attribute name bound in the graph module, real callable).
# graph.py does `from .mcp_tools.X import Y`, so it holds its own reference to
# each tool; rebinding graph.<attr> is what redirects the agent loop the eval
# drives (graph.ask) AND eval_retrieval's direct graph.get_clinvar_record call.
_TOOLS: dict[str, tuple[str, object]] = {
    "clinvar": ("get_clinvar_record", get_clinvar_record),
    "gnomad": ("get_allele_frequency", get_allele_frequency),
    "ensembl": ("get_gene_consequence", get_gene_consequence),
    "ucsc": ("get_genomic_context", get_genomic_context),
    "alphamissense": ("get_alphamissense_score", get_alphamissense_score),
}

# Record-mode rate-limit discipline. NCBI eutils is the tightest (~3 req/s
# unauthenticated); a second of spacing between tools plus between variants keeps
# the whole 37-variant pass well under every source's limit. This is the one-time
# tax -- pay it patiently rather than race and lose rows.
RECORD_SLEEP = 1.0        # seconds between tool calls / between variants
RECORD_RETRIES = 5        # attempts per tool before giving up on a variant
RECORD_BACKOFF = 3.0      # seconds * attempt number, linear backoff on retry


class FixtureMissing(RuntimeError):
    """Raised on replay when a (tool, variant) fixture doesn't exist.

    Deliberately loud: a fixture with a hole reintroduces the row-dropping loss
    the whole record-and-replay layer exists to kill, so a gap must fail the run
    rather than silently score fewer rows.
    """


def _safe(variant: str) -> str:
    """Filename-safe form of a variant id. rsIDs are already safe; this only
    guards against a future HGVS-style id (with ':' '>' '/') landing here."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", variant)


def _fixture_path(tool: str, variant: str) -> Path:
    return FIXTURES / tool / f"{_safe(variant)}.json"


def load_fixture(tool: str, variant: str) -> dict:
    path = _fixture_path(tool, variant)
    if not path.exists():
        raise FixtureMissing(
            f"No {tool} fixture for variant {variant!r} at {path}. "
            f"Record it first: `python evals/run_evals.py --record`."
        )
    return json.loads(path.read_text())


def save_fixture(tool: str, variant: str, payload: dict) -> Path:
    path = _fixture_path(tool, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    # sort_keys + indent so a re-record produces a clean, reviewable git diff.
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _has_all_fixtures(variant: str) -> bool:
    return all(_fixture_path(tool, variant).exists() for tool in _TOOLS)


# --- replay mode ---------------------------------------------------------------

def _replay(tool: str):
    """Build a replay stand-in for `tool`. Accepts and ignores any extra
    positional/keyword args (e.g. ucsc/alphamissense's `consequence=`)."""
    def wrapper(variant: str, *_args, **_kwargs) -> dict:
        return load_fixture(tool, variant)

    wrapper.__name__ = f"replay_{tool}"
    return wrapper


def install() -> None:
    """Rebind the five tool names in the graph module to fixture readers.

    Process-scoped and idempotent. Call once at the start of an offline eval run.
    Leaves src/ on disk and every production importer untouched.
    """
    for tool, (attr, _real) in _TOOLS.items():
        setattr(graph, attr, _replay(tool))


@contextmanager
def replay_tools():
    """Context manager form of install() that restores the originals on exit --
    for tests that want replay for one block without leaking the patch."""
    saved = {attr: getattr(graph, attr) for _, (attr, _real) in _TOOLS.items()}
    install()
    try:
        yield
    finally:
        for attr, fn in saved.items():
            setattr(graph, attr, fn)


# --- record mode ---------------------------------------------------------------

def _call_with_retry(fn, *args, desc: str, **kwargs) -> dict:
    """Call a real tool, retrying on any failure with linear backoff. Retries
    rather than accepting a gap -- the point of the record pass is completeness."""
    for attempt in range(1, RECORD_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - record pass isolates + retries every failure
            if attempt == RECORD_RETRIES:
                raise RuntimeError(f"record failed after {RECORD_RETRIES} attempts: {desc}") from exc
            wait = RECORD_BACKOFF * attempt
            print(f"    ! {desc}: attempt {attempt} failed ({exc}); retrying in {wait:.0f}s")
            time.sleep(wait)
    raise AssertionError("unreachable")  # loop either returns or raises


def record_variant(variant: str, *, sleep: float = RECORD_SLEEP) -> dict:
    """Hit all five real tools for one variant and write their responses to
    fixtures. Ordering mirrors graph.gather_evidence: fetch the Ensembl
    consequence once, then hand it to ucsc and alphamissense as `consequence=`
    so they don't each repeat the VEP round-trip (and so the recorded ucsc /
    alphamissense responses match what the agent actually computes)."""
    results: dict[str, dict] = {}

    results["clinvar"] = _call_with_retry(get_clinvar_record, variant, desc=f"clinvar {variant}")
    time.sleep(sleep)
    results["gnomad"] = _call_with_retry(get_allele_frequency, variant, desc=f"gnomad {variant}")
    time.sleep(sleep)
    ensembl = _call_with_retry(get_gene_consequence, variant, desc=f"ensembl {variant}")
    results["ensembl"] = ensembl
    time.sleep(sleep)
    results["ucsc"] = _call_with_retry(
        get_genomic_context, variant, consequence=ensembl, desc=f"ucsc {variant}"
    )
    time.sleep(sleep)
    results["alphamissense"] = _call_with_retry(
        get_alphamissense_score, variant, consequence=ensembl, desc=f"alphamissense {variant}"
    )

    for tool, payload in results.items():
        save_fixture(tool, variant, payload)
    return results


def record_all(variants: list[str], *, sleep: float = RECORD_SLEEP, force: bool = False) -> None:
    """Record fixtures for every variant. Resumable: variants that already have
    all five fixtures are skipped unless force=True, so a pass interrupted by a
    transient failure can be re-run and pick up where it left off."""
    total = len(variants)
    print(f"Recording fixtures for {total} variant(s) into {FIXTURES}/ (sleep={sleep}s)...")
    for i, variant in enumerate(variants, start=1):
        if not force and _has_all_fixtures(variant):
            print(f"  [{i}/{total}] {variant}: already complete, skipping")
            continue
        print(f"  [{i}/{total}] {variant}: recording 5 tools...")
        record_variant(variant, sleep=sleep)
        if i < total:
            time.sleep(sleep)
    print("Record pass complete. Fixtures frozen.")


def load_variants(limit: int | None = None) -> list[str]:
    with DATASET.open() as f:
        variants = [json.loads(line)["variant"] for line in f if line.strip()]
    return variants[:limit] if limit else variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Record fixtures for the five evidence tools.")
    parser.add_argument("--limit", type=int, default=None, help="only record the first N dataset variants")
    parser.add_argument("--sleep", type=float, default=RECORD_SLEEP, help="seconds between calls (rate-limit tax)")
    parser.add_argument("--force", action="store_true", help="re-record variants that already have fixtures")
    args = parser.parse_args()

    record_all(load_variants(args.limit), sleep=args.sleep, force=args.force)


if __name__ == "__main__":
    main()
