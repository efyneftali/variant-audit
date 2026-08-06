# Temperature vs. voting (VA-41)

_Generated 2026-08-05 · frozen 37-row golden set · k=5 samples/variant · Ollama `llama3.1:8b`._

**Prior going in:** voting wins accuracy, a temperature drop wins cost and latency — pick your poison. **The data rejects the premise.** Temperature=0 wins both.

## The numbers

| Lever | Accuracy | Harm-weighted cost | LLM calls | Latency/variant |
|---|---|---|---|---|
| **temperature=0** (greedy) | **0.865** | **0.324** | **1×** | **~18s** |
| temperature=0.3 | 0.827 mean ± 0.031 SD | 0.400 | 1× | ~18s |
| temperature=0.7 | 0.773 mean ± 0.053 SD | 0.519 | 1× | ~18s |
| best-of-5 voting @ current default temp | 0.757 (majority) | 0.459 | 5× | ~93s |

Best-of-5 voting was measured against the honest single-sample baseline at that same temperature (mean 0.714 ± 0.056 across 5 independent runs, not one lucky draw) — voting recovers real ground over that baseline (+4.3 accuracy points, harm cost 0.546 → 0.459). It just never catches up to temperature=0, despite costing 5× the calls and 5× the latency to get there.

## Decision: keep temperature=0. Drop voting.

Temperature=0 is not a tradeoff against voting on this data — it dominates: highest accuracy, lowest harm-weighted cost, cheapest, fastest. There is no axis on which best-of-5 voting wins outright. Paying 5× to land at 0.757 when 1× already gets 0.865 is not a defensible trade.

**Applied:** `graph.ask()` and `run_evals.py --temperature` now default to `0.0` (previously `None` → unset provider default, which floated near temp≈0.7-0.8 on Ollama — i.e. the *worst* config tested, not the best). Pass an explicit temperature to reproduce the sweep or voting investigation.

## The one caveat — not a hedge, a residual risk to monitor

Temperature=0 is not a strict per-row win. Two rows (`gv-006`, `gv-019`) out of 37 collapse to a stable-wrong answer at temp=0 that non-zero temperature at least sometimes escaped:

- `gv-006` (gold P): majority-vote was *already wrong* at 0.3/0.7 (1/5, 2/5 correct) — temp=0 just removes an already-slim chance. Net neutral vs. voting.
- `gv-019` (gold B): majority-vote was **correct** at both 0.3 and 0.7 (3/5 each) — a best-of-5 vote reliably solves this row. Temp=0 is 0/5, deterministically wrong. Greedy decoding does not have to track the plurality of the sampling distribution, and here it provably didn't.

Neither miss lands in the catastrophic P→B cell (cost 10); both collapse toward the VUS hedge (cost 3 and 1). So the residual risk is real but currently low-severity on this dataset — worth re-checking if the dataset grows or the harm-cost profile of missed rows shifts.

**If this needs revisiting:** the fix is not "raise temperature everywhere" (already shown to lose) — it's targeted resampling only on rows the model's own output signals as uncertain, not a blanket k-sample vote.

_Reproduce: `python evals/run_evals.py --samples 5 --temperature {0.0,0.3,0.7}` (sweep) and `python evals/run_evals.py --samples 5 --temperature 0.7` vs `python evals/run_evals.py --samples 1` (voting cost/latency). Reports: `evals/reports/eval_20260805_{145255,155015,164915,180623,190406}.json`._
