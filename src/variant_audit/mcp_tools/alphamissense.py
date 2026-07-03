"""AlphaMissense tool — DeepMind pathogenicity prediction (missense variants only).

Source: precomputed predictions for ~71M missense substitutions, hosted publicly
on Google Cloud Storage (https://github.com/google-deepmind/alphamissense):
https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz (~613 MiB
gzipped). License is CC BY-NC-SA 4.0 (non-commercial, share-alike) — check this
fits your use case before relying on it beyond research/education.

IMPORTANT: this is a SOTA *computational evidence input*, never ground truth
(grading against it would be grading the model against another model).
Only covers missense — returns a clear "not applicable" for other variant types.

The table is too big to scan per query (~71M rows), so it's loaded once into a
local SQLite database indexed on (chrom, pos, ref, alt). Build it with:

    python -m variant_audit.mcp_tools.alphamissense --build

which downloads the table (if not already present) and builds
data/alphamissense/alphamissense.sqlite. This is a one-time, long-running job —
not something that runs as part of normal dev/test cycles.
"""
from __future__ import annotations

import gzip
import logging
import sqlite3
from pathlib import Path

import requests

from .ensembl import get_gene_consequence

logger = logging.getLogger(__name__)

TABLE_URL = "https://storage.googleapis.com/download/storage/v1/b/dm_alphamissense/o/AlphaMissense_hg38.tsv.gz?alt=media"
DATA_DIR = Path("data/alphamissense")
TSV_GZ_PATH = DATA_DIR / "AlphaMissense_hg38.tsv.gz"
DB_PATH = DATA_DIR / "alphamissense.sqlite"

DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB
BUILD_BATCH_SIZE = 100_000
BUILD_LOG_INTERVAL = 5_000_000


def download_table(dest_path: Path = TSV_GZ_PATH) -> Path:
    """Download the AlphaMissense hg38 table (~613 MiB) if not already present."""
    if dest_path.exists():
        logger.info("alphamissense table already present at %s, skipping download", dest_path)
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    logger.info("downloading AlphaMissense table from %s", TABLE_URL)

    with requests.get(TABLE_URL, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        downloaded = 0
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (50 * DOWNLOAD_CHUNK_SIZE) == 0:
                    logger.info("downloaded %.0f MiB", downloaded / (1024 * 1024))

    tmp_path.rename(dest_path)
    logger.info("download complete: %s", dest_path)
    return dest_path


def build_db(tsv_gz_path: Path = TSV_GZ_PATH, db_path: Path = DB_PATH) -> int:
    """Stream-parse the gzipped table into a SQLite DB indexed on (chrom, pos, ref, alt).

    One-time, long-running (tens of minutes for ~71M rows) — never decompresses
    the whole file to disk, streams it row by row via gzip.open(..., "rt").
    Returns the number of rows loaded.
    """
    if not tsv_gz_path.exists():
        raise FileNotFoundError(f"AlphaMissense table not found at {tsv_gz_path} — call download_table() first")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()  # rebuild from scratch rather than silently appending/duplicating

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute(
        """
        CREATE TABLE variants (
            chrom TEXT NOT NULL,
            pos INTEGER NOT NULL,
            ref TEXT NOT NULL,
            alt TEXT NOT NULL,
            transcript_id TEXT,
            protein_variant TEXT,
            am_pathogenicity REAL,
            am_class TEXT
        )
        """
    )

    row_count = 0
    batch: list[tuple] = []
    logger.info("building AlphaMissense DB from %s", tsv_gz_path)

    with gzip.open(tsv_gz_path, "rt") as f:
        for line in f:
            if line.startswith("#") or line.startswith("CHROM\t"):
                continue
            chrom, pos, ref, alt, _genome, _uniprot_id, transcript_id, protein_variant, am_pathogenicity, am_class = (
                line.rstrip("\n").split("\t")
            )
            batch.append((chrom, int(pos), ref, alt, transcript_id, protein_variant, float(am_pathogenicity), am_class))
            row_count += 1

            if len(batch) >= BUILD_BATCH_SIZE:
                conn.executemany("INSERT INTO variants VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
                conn.commit()
                batch.clear()
                if row_count % BUILD_LOG_INTERVAL < BUILD_BATCH_SIZE:
                    logger.info("loaded %d rows...", row_count)

    if batch:
        conn.executemany("INSERT INTO variants VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
        conn.commit()

    logger.info("creating index on (chrom, pos, ref, alt)...")
    conn.execute("CREATE INDEX idx_variant_lookup ON variants (chrom, pos, ref, alt)")
    conn.commit()
    conn.close()

    logger.info("done: %d rows loaded into %s", row_count, db_path)
    return row_count


def get_alphamissense_score(variant: str, consequence: dict | None = None) -> dict:
    """Look up a missense variant's AlphaMissense pathogenicity score.

    Args:
        variant: an rsID, e.g. "rs28897696".
        consequence: an already-fetched get_gene_consequence(variant) result, to
            avoid a second VEP round-trip when the caller already has one (e.g.
            graph.gather_evidence). Fetched internally if omitted.
    Returns:
        {variant, found, applicable, am_pathogenicity, am_class, transcript_id,
         protein_variant}. `applicable` is False for any non-missense variant
        (AlphaMissense doesn't cover them) or when Ensembl couldn't resolve the
        variant at all — `found` is always False when `applicable` is False.
        When applicable but this exact substitution isn't in the precomputed
        table, found=False, applicable=True.

    Raises:
        RuntimeError: if Ensembl's own lookup fails (see get_gene_consequence).
        FileNotFoundError: if the local SQLite DB hasn't been built yet — run
        `python -m variant_audit.mcp_tools.alphamissense --build` first.
    """
    if consequence is None:
        consequence = get_gene_consequence(variant)

    if not consequence.get("found") or consequence.get("most_severe_consequence") != "missense_variant":
        return {"variant": variant, "found": False, "applicable": False}

    chrom, pos, ref, alt = consequence.get("chrom"), consequence.get("start"), consequence.get("ref"), consequence.get("alt")
    if chrom is None or pos is None or not ref or not alt:
        return {"variant": variant, "found": False, "applicable": True}

    chrom = str(chrom) if str(chrom).startswith("chr") else f"chr{chrom}"
    row = _query_db(chrom, pos, ref, alt, variant)
    if row is None:
        return {"variant": variant, "found": False, "applicable": True}

    transcript_id, protein_variant, am_pathogenicity, am_class = row
    return {
        "variant": variant,
        "found": True,
        "applicable": True,
        "am_pathogenicity": am_pathogenicity,
        "am_class": am_class,
        "transcript_id": transcript_id,
        "protein_variant": protein_variant,
    }


def _query_db(chrom: str, pos: int, ref: str, alt: str, variant: str) -> tuple | None:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"AlphaMissense DB not found at {DB_PATH} — "
            "run `python -m variant_audit.mcp_tools.alphamissense --build` first"
        )
    conn = sqlite3.connect(DB_PATH)
    try:
        logger.info("alphamissense lookup variant=%s chrom=%s pos=%d ref=%s alt=%s", variant, chrom, pos, ref, alt)
        cursor = conn.execute(
            "SELECT transcript_id, protein_variant, am_pathogenicity, am_class "
            "FROM variants WHERE chrom = ? AND pos = ? AND ref = ? AND alt = ? LIMIT 1",
            (chrom, pos, ref, alt),
        )
        return cursor.fetchone()
    finally:
        conn.close()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if "--build" in sys.argv:
        download_table()
        build_db()
    else:
        print(__doc__)
