# Golden Dataset — construction methodology & scoring rubric

*This doc is the credible-evals story. The dataset isn't "30 questions I made up" — it's a deliberately constructed, stratified, versioned set anchored to authoritative ground truth. Build it on Day 11; document every decision here.*

> The 3 rows in `golden_dataset.jsonl` are **schema examples**, not the real set. Replace them.

## Source of ground truth
ClinVar `variant_summary.txt.gz` (bulk download — no VCF parsing). Columns you'll use: HGVS name, gene, clinical significance, **review status** (gold stars), molecular consequence, allele frequency. ClinVar is the *answer key*; in the agent it's just one evidence input (don't grade the model against itself).

## Schema (one JSON object per line)
| field | meaning |
|---|---|
| `id` | stable identifier |
| `variant` | canonical HGVS (normalize representation) |
| `gene` | gene symbol |
| `gold_label` | collapsed 3-class: `P` / `VUS` / `B` |
| `clinvar_significance` | raw ClinVar label (provenance) |
| `review_stars` | 0–4, ClinVar review confidence (difficulty proxy) |
| `evidence_type` | dominant ACMG evidence (frameshift/LoF, population_frequency, missense/computational, functional, conflicting…) |
| `difficulty` | `easy` / `hard` / `hardest` |
| `expected_behavior` | `classify` or `abstain` |

## Stratification (so an accuracy number means something)
- **Difficulty** via review stars + call type: easy = 2–3★ unambiguous P/B; hard = 1★ or near-threshold; hardest = VUS / conflicting.
- **Gene / mechanism**: spread across genes with different mechanisms (LoF genes like BRCA1 exercise PVS1; missense-driven genes exercise different logic).
- **Evidence type**: pick variants whose call rests on different ACMG criteria, so the agent's reasoning is tested across evidence kinds, not one.
- Target ~30–50 examples (trim to 30 if behind — see plan cut-list).

## Defining "correct" (your domain judgment shows here)
- **Collapse to 3 clinically-actionable classes** (P+LP→P, B+LB→B, VUS). LP-vs-P disagreement doesn't change management — don't penalize it.
- **Asymmetric scoring**: calling a truly *pathogenic* variant *benign* is the catastrophic error; weight errors by clinical harm, not flat accuracy.
- **VUS = abstention test**: correct behavior on a true VUS is to return VUS / not produce a confident P or B. Penalize a confident wrong call on a VUS heavily.

## Adversarial / abstention subset (hand-curated — your moat)
- **ClinVar "conflicting interpretations"** variants → correct behavior is uncertain, not a confident pick.
- **Insufficient-context** cases → the relevant criterion isn't in the corpus → expect "insufficient evidence."
- **Look-alike traps** → looks pathogenic by position but benign by frequency → tests use of population data.

## Synthetic augmentation (Day 11)
Generate hard/edge cases programmatically alongside the real anchor — demonstrates data augmentation. Keep synthetic clearly tagged and separate from real-data ground truth.

## Versioning
Treat the dataset like code: version it, log how each batch was sampled, never silently edit. Score changes are only meaningful against a fixed dataset.
