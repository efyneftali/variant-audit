# Groundedness judge calibration set (VA-35, step 4)

Before you trust the LLM groundedness judge (`graph.check_grounded`), you have to
know how often it agrees with a human expert. That means a **gold set of human
labels** to score the judge against — and those labels have to be *yours*, made
*blind*, before you see any model's answer. This directory is the harness for
producing that set; the labels are the one part that can't be automated, because
the whole point is to capture your genetics judgment rather than a model's.

## Why blind, and why hand-labeled

- **Blind** — if you label after seeing the judge's verdict, you anchor on it:
  borderline cases drift toward whatever the model said, and the agreement number
  you later compute is inflated and meaningless. The cases you label contain **no
  verdict**, and the labeler never runs or imports the judge, so blindness is
  structural, not willpower.
- **Hand-labeled by you** — groundedness of an ACMG classification is a genetics
  judgment (does BA1 really apply at this frequency? is PVS1 defensible for this
  consequence? is the tier supported by *these* five sources?). An automated
  "pull" of labels — from ClinVar stars, from another model — would just relabel
  the problem with the same kind of system you're trying to audit. This is the
  step where your background is the differentiator.

## The workflow

```
# 1. Build the cases (needs Ollama/Anthropic + Qdrant, like a normal eval run;
#    evidence tools are replayed from fixtures, so record those first if needed).
python evals/calibration/build_cases.py            # ~37 real + 8 injected = ~45 cases

# 2. Label them blind. Resumable — stop with `q`, rerun to continue.
python evals/calibration/label.py --labeler efy

# 3. (recommended) A second independent pass a day later, to measure your own
#    self-consistency (intra-rater agreement). Different file, different order:
python evals/calibration/label.py --out labels_pass2.jsonl --seed 99 --labeler efy
```

Aim for **30–50 labeled cases**. Below ~30 the agreement estimate is too noisy to
act on; past ~50 you hit diminishing returns for a judge this scoped.

## What a "case" is

Exactly what the judge sees: a variant, the classification text the agent
produced, and the evidence + retrieved ACMG criteria it was given. Cases are
frozen from the offline fixtures so they're reproducible — the artifact you label
today is byte-identical to the one the judge is scored against later.

Two provenances (tracked in a `_provenance` field you never see while labeling):
- **real** — a genuine agent classification. The natural distribution, mostly
  grounded.
- **injected_fabricated_code** — a real case with a citation to an ACMG code that
  is *provably absent* from the retrieved criteria (known-ungrounded by
  construction). These exist only so the minority class is represented well
  enough to measure agreement; they carry an `intended: "ungrounded"` marker so a
  later step can confirm you agreed. Set `--injected 0` to build real cases only.

## How to label (the rubric)

A classification is **ungrounded** if either:
1. it **cites an ACMG code that does not appear** in the retrieved criteria text
   (an invented citation), or
2. it **asserts a fact the evidence doesn't support** (e.g. "absent from gnomAD"
   when the frequency block shows a value).

Otherwise it's **grounded**. Two things you are explicitly *not* judging:
- **not** whether the call is *clinically correct* — a wrong-but-supported call is
  still grounded; groundedness is about faithfulness to the given inputs.
- **not** the tier name — all five ACMG tiers are valid labels.

Write a one-line reason for every case, and for ungrounded ones name the offending
criterion or claim. Those reasons are what make disagreements with the judge
diagnosable later, not just countable. If a case is genuinely a coin-flip, `s`kip
it rather than force a label — a smaller clean set beats a padded noisy one.

## Files

| File | What it is |
|------|------------|
| `build_cases.py` | freezes cases from fixtures; writes `cases.jsonl` (no labels, no verdicts) |
| `label.py` | blind labeler; appends to `labels.jsonl` (resumable) |
| `cases.jsonl` | the frozen cases (generated) |
| `labels.jsonl` | your labels: `{id, variant, label, rationale, offending_criterion, labeler, labeled_at}` |

## Don't train on the test set — keep a holdout

Tuning the judge prompt against the same cases you then report κ on is training
on the test set: you'd be fitting the prompt to this specific 45, and the number
would come out optimistically biased. Same rigor the dataset gets.

So freeze a **dev/test split** once, right after labeling:

```
python evals/calibration/make_split.py        # 40% holdout, stratified by your label, frozen to split.json
```

Then the discipline is:
- **Inspect disagreements and edit the judge prompt against the `dev` split only.**
- **Report the headline κ on the `test` split** — cases the prompt has never seen.
- The split is frozen in `split.json` and stratified by your label (both classes
  in each partition). It refuses to silently re-roll — re-rolling until the number
  looks good is the same cheating in a different spot.

If you *don't* keep a holdout, that's allowed — but then the report says so, in
capitals: scoring `--split all` stamps the number **IN-SAMPLE** and calls it an
upper bound. What's not allowed is quietly tuning on all 45 and reporting κ as if
it were held out.

## Score the candidate judges (the payoff)

`score_judges.py` runs each candidate judge over the cases, computes **Cohen's κ**
(judge vs. you), and writes `JUDGE_CALIBRATION.md` with the table, the pick, and a
banner saying whether the κ is holdout or in-sample.

```
python evals/calibration/score_judges.py --split test --candidates local,haiku,sonnet   # holdout (report this)
python evals/calibration/score_judges.py --split dev  --candidates haiku                 # iterate the prompt here
python evals/calibration/score_judges.py --split all  --candidates local                 # in-sample (stamped as such)
python evals/calibration/score_judges.py --threshold 0.6                                  # your agreement bar
```

With a frozen split, `--split` defaults to `test`. Without one, it defaults to
`all` and stamps the report IN-SAMPLE.

**Cost.** `local` is free; `haiku` and `sonnet` are paid Anthropic calls, so
they're **opt-in** — the default runs only `local`. The bill is tiny: at most one
call per labeled case per candidate (the deterministic pre-check settles the clear
ones for free), so ~45 cases is well under $0.15 for Haiku and roughly $0.30–0.40
for Sonnet at current prices. Every raw verdict is cached in `judge_runs/`, so
re-scoring or moving the threshold later costs nothing — `--force` re-pays only
when you actually want a fresh run.

The pick is the **cheapest candidate that clears your κ threshold** (default 0.6,
"substantial agreement") — no reason to pay for Sonnet if Haiku already agrees
with you. The report also splits the judge's errors into `false_ungrounded`
(over-strict — flags a grounded call) vs. `missed_ungrounded` (dangerous — passes
a real problem), because those cost you very differently.

Keep `cases.jsonl`, `labels.jsonl`, `judge_runs/`, and `JUDGE_CALIBRATION.md`
under version control so the whole comparison is reproducible.

## Wiring it back into the eval harness

`evals/run_evals.py::eval_generation_judge` is still a stub. It should run the
selected judge over these cases and report the same κ, so a future prompt change
to the judge is regression-checked against your labels — the calibration set is
the fixed ruler, `score_judges.py` is the one-off bake-off.
