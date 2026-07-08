# Golden Dataset — construction methodology & scoring rubric

*The dataset isn't "30 questions I made up" — it's a deliberately constructed, stratified, versioned set anchored to authoritative ground truth.*

## Source of ground truth
ClinVar `variant_summary.txt.gz` (bulk download, no VCF parsing), downloaded 2026-07-08 from `https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz`. ClinVar is the *answer key*; in the agent it's just one evidence input (`mcp_tools/clinvar.py`) — the eval never grades the model against the same signal it was given, since ground truth is fixed offline and the agent independently gathers evidence at query time.

## Schema (one JSON object per line)
| field | meaning |
|---|---|
| `id` | stable identifier (`gv-NNN` real anchor, `syn-NNN` synthetic) |
| `variant` | **rsID** (e.g. `rs28897696`) — this is what's actually passed to `ask()`/`gather_evidence`. Four of the five MCP tools (gnomAD, Ensembl/VEP, UCSC, AlphaMissense) accept only an rsID, not HGVS, so this field must be a resolvable rsID or the agent can't run at all. |
| `hgvs_name` | ClinVar's HGVS name, for human readability/provenance only — never fed to a tool |
| `gene` | gene symbol |
| `gold_label` | collapsed 3-class: `P` / `VUS` / `B` |
| `clinvar_significance` | raw ClinVar label (provenance) |
| `review_stars` | 0–4, ClinVar review confidence (difficulty proxy) |
| `evidence_type` | dominant ACMG evidence (`frameshift/LoF`, `population_frequency`, `missense/computational`, `conflicting_evidence_sources`) |
| `difficulty` | `easy` / `hard` / `hardest` |
| `expected_behavior` | `classify` or `abstain` |
| `source` | `clinvar` (real anchor) or `synthetic` (rule-mined, see below) |
| `allele_freq` | gnomAD allele frequency, present where frequency was load-bearing evidence |
| `vep_most_severe_consequence` | Ensembl VEP's call, recorded as proof the rsID was live-validated at build time |
| `synthetic_rule` | which mining rule produced a synthetic row (only on `source: synthetic` rows) |

## Build pipeline (`evals/build/`)
Scripts are kept for reproducibility; `variant_summary.txt`/`.txt.gz` are gitignored (too large, re-downloadable).

1. **`extract_candidates.py`** — streams all ~9M rows, filters to `Assembly == GRCh38` (the file has one row per variant per assembly; GRCh37 rows are exact duplicates and would double-count), requires a real rsID (`RS# (dbSNP) != -1` — the hard technical constraint above), dedupes by `VariationID`, computes review stars from the free-text `ReviewStatus` column (ClinVar doesn't store stars as a number), and collapses `ClinicalSignificance` to `P`/`B`/`VUS` (conflicting-interpretation rows are kept, unlabeled, for the abstention pool; out-of-scope categories like risk-factor/drug-response/association are dropped — they don't fit the 3-class scheme).
   - Funnel: 8,992,046 rows read → 4,462,274 GRCh38 → 2,896,843 with rsID → 2,881,532 in-scope after dedup.
2. **`select_pool.py`** — narrows to a curated 26-gene list spanning three mechanisms (so different ACMG criteria actually get exercised): LoF/PVS1 genes (BRCA1, BRCA2, MLH1, MSH2, MSH6, PMS2, APC, PTEN, TP53, LDLR, VHL, NF1, RB1), missense/computational genes (MYH7, KCNQ1, SCN5A, RYR1, COL1A1, FBN1, GJB2), and common-variant/frequency genes (CFTR, HFE, MTHFR, F5, SERPINA1, APOE). Buckets by difficulty tier.
3. **`sample_and_validate.py`** — samples a target composition (12 easy / 12 hard / 12 hardest, split across gold labels and evidence types) and **live-validates every candidate against the pipeline's own `mcp_tools.ensembl.get_gene_consequence`** before accepting it — an rsID existing in ClinVar isn't sufficient, since `graph.py`'s `gather_evidence` calls Ensembl VEP unconditionally and a resolution failure there breaks the whole run. Candidates are dropped (and the next one in their stratum tried) if VEP doesn't resolve the rsID, or resolves it to a different gene than ClinVar's record names.
   - For `population_frequency`-tagged rows specifically, `allele_freq` from `mcp_tools.gnomad` is required to be **> 1%** before acceptance. First pass without this gate accidentally selected several "Benign" ClinVar calls that were actually just rare synonymous changes (frequency near zero) — benign for a different reason (no functional impact, not population commonality). 30 candidates were correctly rejected on the re-run before all 6 population-frequency slots were filled with genuinely common variants (3.6%–30% allele frequency).
   - For `missense/computational` rows, `mcp_tools.alphamissense` was queried for a score; 14 of 19 came back without one because the local AlphaMissense prediction DB (`data/alphamissense/alphamissense.sqlite`) hasn't been built in this environment — evidence-type tagging is unaffected (it's driven by VEP consequence), but the score itself isn't stored here.
4. **`mine_synthetic.py`** — see below.
5. **`assemble_final.py`** — merges the validated real anchor + synthetic rows into `golden_dataset.jsonl`, sorted by difficulty then label then gene.

## Stratification (so an accuracy number means something)
- **Difficulty** via review stars + call type: easy = 2★+ unambiguous P/B; hard = 0–1★ single-submitter P/B; hardest = confidently-called VUS (2★+) or conflicting-interpretation calls.
- **Gene / mechanism**: 22 distinct genes used across the 37 rows, capped at 3 uses of any single gene, spanning the three mechanism families above.
- **Evidence type**: `frameshift/LoF` (11), `missense/computational` (19), `population_frequency` (6), `conflicting_evidence_sources` (1) — driven by each variant's actual VEP consequence and gnomAD frequency, not just which gene list it came from.

## Defining "correct"
- **Collapse to 3 clinically-actionable classes** (P+LP→P, B+LB→B, VUS). LP-vs-P disagreement doesn't change management — don't penalize it.
- **Asymmetric scoring**: calling a truly *pathogenic* variant *benign* is the catastrophic error; weight errors by clinical harm, not flat accuracy.
- **VUS = abstention test**: correct behavior on a true VUS (confidently-called or conflicting) is to return VUS / not produce a confident P or B. Penalize a confident wrong call on a VUS heavily.

## Adversarial / abstention subset (hand-curated — the moat)
6 of the 37 rows are `expected_behavior: abstain` — real ClinVar "Conflicting interpretations of pathogenicity" calls across BRCA1/2, APC, MSH2/6, PMS2, NF1, RYR1, MYH7, SCN5A, FBN1, COL1A1, TP53, LDLR, VHL and others. Correct behavior is an uncertain call, not a confident pick.

## Synthetic augmentation (rule-mined, not fabricated)
Every tool in this pipeline requires a real, resolvable rsID — an invented variant would simply fail to run, so "synthetic" here means **programmatically selected real ClinVar data**, not fabricated ground truth. `mine_synthetic.py` scans the full 2.88M-row in-scope pool (not just the curated 26-gene list) for a specific adversarial pattern via an automated rule: a variant ClinVar calls Pathogenic/Likely pathogenic on a single submitter's say-so (0–1★, non-conflicting), where gnomAD shows the variant is actually common in the population (allele frequency > 1%) — a direct contradiction, since BA1/BS1 says a variant this common can't be causing a rare Mendelian disorder.

One hit survived after checking 250 randomly-sampled candidates (gnomAD's public API rate-limits heavily under sustained load, so most of the budget went to retries, not genuine misses): **`rs61735712` (KCNC2)** — ClinVar: "Pathogenic", 0★, no assertion criteria; gnomAD: 7.3% allele frequency. Tagged `source: synthetic`, `gold_label: VUS`, `expected_behavior: abstain`, `synthetic_rule: look_alike_trap_frequency_vs_label`. Kept separate from the real-anchor accuracy numbers (filter on `source` in the harness) since its label reflects a rule-based judgment call (frequency contradicts the assertion), not a ClinVar-adjudicated VUS.

## Versioning
- Built 2026-07-08 against the ClinVar release fetched that day (see `evals/build/extract_candidates.py` docstring for the exact URL).
- Total: **37 rows** (36 real ClinVar anchor + 1 rule-mined synthetic) — within the 30–50 target.
- Composition: difficulty 12 easy / 12 hard / 13 hardest; gold label 12 P / 12 B / 13 VUS; expected behavior 30 classify / 7 abstain.
- Treat the dataset like code: version it, log how each batch was sampled (build scripts + logs kept in `evals/build/`), never silently edit. Score changes are only meaningful against a fixed dataset — bump a version note here if `golden_dataset.jsonl` is regenerated.
