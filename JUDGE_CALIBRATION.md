# Judge calibration (VA-35)

_Generated 2026-08-03 23:09 UTC · threshold κ ≥ 0.60 (substantial agreement, Landis & Koch)._

**Holdout κ** on the `test` split (18 cases the judge prompt was never tuned against). Dev-split cases are excluded — tune the prompt there, report here.

Cohen's κ is agreement between each candidate judge and the human labels, corrected for chance. `false_ungrounded` = judge flagged a grounded call (over-strict); `missed_ungrounded` = judge passed a real problem (the dangerous error).

| candidate | provider / model | n | accuracy | κ | false_ungrounded | missed_ungrounded | clears? |
|-----------|------------------|---|----------|---|------------------|-------------------|---------|
| haiku (paid) | `anthropic/claude-haiku-4-5-20251001` | 18 | 0.889 | **0.727** | 2 | 0 | ✅ |
| local | `ollama/llama3.1:8b` | 18 | 0.611 | **0.323** | 0 | 7 | — |

## Selected judge: **haiku** (κ = 0.727)

`anthropic/claude-haiku-4-5-20251001` — the cheapest candidate clearing κ ≥ 0.60. Set it as the judge with `JUDGE_PROVIDER=anthropic` and `ANTHROPIC_JUDGE_MODEL=claude-haiku-4-5-20251001`.

_Reproduce: `python evals/calibration/score_judges.py --candidates local,haiku,sonnet --split test`. Raw per-case verdicts are cached in `evals/calibration/judge_runs/`._
