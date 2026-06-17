# EvalForge — Project Spec (one page)

## One-liner
A clinical variant-classification assistant — and the eval harness that decides whether it's safe to ship. The assistant is the *system under test*; **the harness is the headline.**

![Variant Audit — overview](variant_audit_overview.png)

## The problem it showcases
Most people can build a RAG demo. Almost no one builds the thing that tells you whether the demo is *right*. EvalForge demonstrates the harder, scarcer discipline — calibrated evaluation, statistically honest comparison, CI quality gates, and a production failure-mining loop — on a domain where ground truth is authoritative and **I can personally verify correctness**: clinical genetics.

## What it does
Given a genetic variant ("Is *BRCA1* c.68_69delAG pathogenic, and why?"), the system gathers evidence, reasons through the ACMG classification rules, and returns a classification *with citations* ("Likely Pathogenic — absent from population databases (PM2); frameshift in a loss-of-function gene (PVS1)"). The harness then runs hundreds of variants against ClinVar's known answers and reports how often it's right, where it fails, and whether the result is trustworthy.

## The three AI primitives (and why each is here)
- **RAG = the rulebook.** Corpus is the ACMG/ClinGen classification guidelines. Retrieval pulls the criteria relevant to a variant. The system reasons from the *published standard*, not model memory.
- **MCP = the live evidence.** Variant-specific facts live in databases that update constantly. Each is an MCP tool: `get_clinvar_record()`, `get_allele_frequency()` (gnomAD), `get_gene_consequence()` (Ensembl/VEP), `get_genomic_context()` (UCSC REST API — conservation, genes, regulatory regions), `get_alphamissense_score()` (DeepMind AlphaMissense — SOTA computational evidence, missense only, an input not ground truth). MCP is the standardized protocol exposing them as callable tools — reusable by the app, the harness, or any future agent.
- **State machine (LangGraph) = the reasoning workflow.** Classification is multi-step with branches: gather evidence → grade sufficiency → (loop back for more, bounded) → classify → check grounded → (reclassify, bounded) → return. Bounded loops guarantee termination — the genuinely senior design point.

## Architecture diagram

![Variant Audit — detailed architecture](variant_audit_architecture.png)

*Assistant (blue) = the LangGraph state machine under test. Amber = MCP tools to live databases. Green = the RAG knowledge corpus and ClinVar ground truth. Purple = the eval harness wrapping it all. Dashed edges = bounded correction loops and the failure-mining feedback loop.*

## The harness (the differentiator — 4 layers)
1. **Dataset engineering** — versioned golden set, stratified by difficulty, tagged by expected failure mode, including adversarial/abstention cases where correct behavior is "uncertain."
2. **Judge calibration** — LLM-as-judge measured against my own hand-labels (Cohen's kappa), prompt tuned until agreement is acceptable.
3. **Statistical rigor** — N runs, reported variance, honest A/B comparison so a 3-point delta isn't mistaken for signal.
4. **CI + production loop** — evals gate every PR; nightly eval CronJob alerts on regression; failure-mining feeds production misses back into the golden set.

## Scope boundaries
- **In:** the assistant (MCP + RAG + state machine), the harness (all 4 layers), containerization, local Kubernetes (kind) + Prometheus/Grafana, CI.
- **Out / optional:** EKS cloud deploy, async eval workers. These are *additive breadth*, never load-bearing. (See cut-list in `EvalForge_4week_plan.md`.)
- **v2 (post-core):** a second agent — a primer-design assistant with objective, computable ground truth — evaluated by the *same* harness, turning the system into a general eval *platform*. (See v2 section in the plan.)
- **Explicitly not building:** VCF parsing or any bioinformatics plumbing. Ground truth comes from ClinVar IDs directly.

## Deliverables (what makes it real and talkable)
- Public repo with README written for a staff engineer + architecture diagram
- `golden_dataset.jsonl` + `DATASET.md` documenting construction methodology
- `JUDGE_CALIBRATION.md` with a measured kappa
- A scorecard history showing one real, data-driven decision
- A "3 production failure modes I found and fixed" writeup
- Grafana dashboard separating platform health from answer quality

## Success criteria
A stranger can read the repo and understand it; I can whiteboard and defend the whole system in 45 minutes across two tracks — **AI architecture** (MCP + RAG + state machine) and **eval rigor** (the harness) — without underselling once.

## Stack
Python · LangGraph · Qdrant (Docker/local) · sentence-transformers (local embeddings) · Claude API (generation/judging) · FastAPI · OpenTelemetry + Prometheus + Grafana · GitHub Actions · Kubernetes (kind locally; EKS optional)

## Tech ↔ data map
| Need | Source | Mechanism |
|---|---|---|
| Classification rules | ACMG/ClinGen guideline text | RAG |
| Live variant facts | ClinVar, gnomAD, Ensembl, UCSC, AlphaMissense | MCP tools |
| Ground-truth labels | ClinVar (free, authoritative) | Eval harness |

---
*Companion docs in this folder: `EvalForge_4week_plan.md` (day-by-day execution) · `CONTENT_SYSTEM.md` (documenting the journey).*
