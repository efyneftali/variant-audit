# EvalForge — 4-Week Full-Time Build Plan

Pairs with **EvalForge_SPEC.md** (what the project is) and **CONTENT_SYSTEM.md** (documenting the journey). This is the day-by-day execution guide.

**Goal = interview-readiness,** not "finished software": a clean public repo and the ability to whiteboard and defend the system cold, across two tracks — AI architecture (MCP + RAG + state machine) and eval rigor (the harness).

**Assumptions:** ~7 hrs/day deep build, Mon–Fri, weekends as buffer/rest. Plus a daily ~1.5–2 hr job-search block and a 30-min system-design drill (you're interviewing in parallel).

---

## Operating principles
- **Define "done" before you start each day.** A specific exit condition, not "work on the harness." Kills scope creep.
- **Depth-first.** The harness (Week 3) is the edge. Slips come out of Week 4 (breadth), never Week 3.
- **Always have a runnable thin slice.** End of Week 1 the system answers one question end-to-end; everything after deepens a working thing.
- **You drive, I assist.** You articulate the goal and attempt it in your own words first; I refine and unblock. (See "How we work together," bottom.)
- **Job search is parallel and non-negotiable.** Network from week 1; real applications start end of week 1.

---

## WEEK 1 — Foundations + thin end-to-end slice
**Exit gate:** one variant in → classification + citation out, running end-to-end.

- **Day 0 (half day) — Setup.** `brew install colima docker`; venv; repo; Anthropic key in `.env`; Qdrant in Docker. *Done:* one successful API call + Qdrant reachable.
- **Day 1 — Anthropic API.** Multi-turn, streaming, tool use (full two-turn flow). *Done:* Claude calls a Python function you wrote and uses the result.
- **Day 2 — RAG from scratch.** Local embeddings; chunk + embed + upsert an ACMG guideline doc; query. *Done:* `search "PVS1 loss of function"` returns the right passage.
- **Day 3 — RAG → cited answer + API skeleton.** Retrieval into a generate call with citations; FastAPI `/ask` skeleton. *Done:* POST a question, get a cited answer grounded in the corpus.
- **Day 4 — MCP basics.** Read the spec; one MCP server, one tool `get_clinvar_record()` hitting real ClinVar. *Done:* Claude/MCP Inspector calls your tool, gets a real record.
- **Day 5 — Thin slice + self-demo.** Linear flow: variant → ClinVar tool → retrieve criteria → classify. README stub. *Done:* one variant → classification + citation, top to bottom.

*SD drill:* redraw a past pipeline/notification/lineage system from memory; start Alex Xu Vol 1.

---

## WEEK 2 — The genomics system (state machine + MCP tools)
**Exit gate:** 5 hand-picked variants classify sensibly with citations, via the LangGraph state machine with bounded loops.

- **Day 6 — LangGraph fundamentals.** Port the slice to a `StateGraph` (retrieve → generate → END), typed state. *Done:* same output as Day 5, now through LangGraph.
- **Day 7 — More MCP tools (richer evidence).** Add `get_allele_frequency()` (gnomAD), `get_genomic_context()` (UCSC REST API `api.genome.ucsc.edu` — conservation/phyloP, nearby genes, regulatory regions; no auth, 1 req/sec), and `get_alphamissense_score()` (DeepMind AlphaMissense, served from the downloaded Zenodo table — a SOTA computational-evidence source, *missense only*, an input not ground truth). Rate limiting, error handling, strict schemas. *Done:* tools return structured results; a bad/unknown variant is handled gracefully.
- **Day 8 — Evidence + grading nodes.** `gather_evidence` calls the tools; `grade_evidence` judges sufficiency. *Done:* graph gathers from all sources and judges sufficiency.
- **Day 9 — Branching + bounded loops.** Conditional edges: insufficient → gather more (bounded); classify → check-grounded → reclassify (bounded). *Done:* a thin-evidence variant visibly triggers the loop; everything terminates.
- **Day 10 — Classification + tag v0.1.** `classify` combines ACMG criteria into Pathogenic/Benign/VUS with cited criteria. *Done:* 5 variants classify sensibly; tag `v0.1`.

*SD drill:* "design a data pipeline" and "design a notification system" cold, timed, narrated.

---

## WEEK 3 — The eval harness  ★ THE HEADLINE — protect this week ★
**Exit gate:** a real accuracy number, a calibrated judge (measured kappa), one data-driven decision, CI that fails on regression.

- **Day 11 — Golden dataset.** 30–50 ClinVar-labeled triples, stratified by difficulty/gene/evidence type, tagged by failure mode, ~5 adversarial/abstention cases. **Augment with synthetic edge cases** (programmatically generated hard examples) alongside the real-data anchor — demonstrates data augmentation. *Done:* versioned `golden_dataset.jsonl` + `DATASET.md` documenting the sampling logic and scoring rubric.
- **Day 12 — Metrics + runner.** Accuracy vs ClinVar, recall@k, per-failure-mode breakdown, JSON report. *Done:* `make eval` prints a scorecard with a real accuracy number.
- **Day 13 — Judge + calibration.** Faithfulness/groundedness judge; hand-label 50 outputs; measure judge-vs-you agreement (kappa); tune. *Done:* `JUDGE_CALIBRATION.md` with a kappa.
- **Day 14 — Statistical rigor + robustness.** N runs, variance/confidence; an A/B compare (e.g., chunk 256 vs 512), make ONE decision from the data. **Add an adversarial-robustness check:** perturb inputs (paraphrase, noise, odd formatting) and measure whether quality holds — demonstrates robustness/red-team instinct. *Done:* a documented decision backed by an honest comparison + a robustness number.
- **Day 15 — CI release-gating pipeline.** GitHub Actions runs evals on every PR / prompt change / model bump; fails if accuracy/recall drops past threshold. *Done:* a PR that degrades quality goes red in CI.

*SD drill:* design an observability system and a rate limiter; then one unfamiliar problem (search index / URL shortener).

---

## WEEK 4 — Breadth + ship + interview ready
**Exit gate:** repo public and clean; runs on Kubernetes with Prometheus SLOs; full 45-min mock done without underselling.

- **Day 16 — Containerize + local K8s.** Multi-stage Dockerfiles (API + MCP server, non-root); deploy to `kind` with kube-prometheus-stack. *Done:* system runs on a local cluster, Prometheus scraping it.
- **Day 17 — Observability.** App `/metrics` (per-node latency, token spend, eval pass rates); Grafana dashboard separating **platform health** from **answer quality**. *Done:* dashboard shows both, clearly distinguished.
- **Day 18 — Eval CronJob + production loop.** Nightly eval CronJob → Prometheus alert on regression; failure-mining script feeds misses back to the dataset to keep evals **unsaturated/high-signal**. *(Stretch: a simple CLI/UI so a non-engineer could author + run an eval; async workers + queue.)* *Done:* nightly eval runs; a simulated regression fires an alert.
- **Day 19 — Ship + write-up.** README for a staff engineer; architecture diagram; "3 production failure modes I found and fixed"; clean secrets/env; push public. *Done:* repo public, clean, demo-able by a stranger.
- **Day 20 — Interview ready.** Drill both narratives until automatic; full mock (behavioral + technical + system design, design this system cold); update LinkedIn. *Done:* defend the whole thing in 45 min without underselling.

*SD drill:* full 45-min mock whiteboards; this system is your anchor problem.

---

## Cut-list (if behind — cut from the TOP first)
1. Async eval workers (Day 18 stretch) — describe verbally instead.
2. EKS / cloud deploy — `kind` locally is enough to demo and talk about.
3. Third MCP tool (Ensembl) — two live tools prove the pattern.
4. Trim the golden dataset to 30 examples.

**Never cut:** Week 3 (the harness). Rough everything else but a rigorous, calibrated harness = you still have the differentiated project. The reverse isn't true.

---

## Tailoring for agent-evaluation / trust & safety roles

This project maps almost 1:1 onto agent-eval roles (e.g., Anthropic's safety-systems eval infra). Use their vocabulary when you describe it — the work is the same, the words should match the posting:

| Their language | Where it lives in this build |
|---|---|
| "eval harness for a complex long-horizon agent" | the harness around your LangGraph state machine (multi-step, tool-using) |
| "datasets representing real-world misuse, not synthetic benchmarks" | ClinVar-anchored real-data dataset (+ synthetic *augmentation*, Day 11) |
| "detection precision/recall, robustness" | accuracy, asymmetric error cost, abstention + robustness check (Day 14) |
| "run on every agent change, prompt update, model upgrade" | CI release-gating pipeline (Day 15) + judge re-calibration on model bump (Day 13) |
| "keep evals unsaturated and high-signal" | failure-mining loop (Day 18) |
| "tooling so policy experts author evals without engineering" | non-engineer author CLI/UI (Day 18 stretch) |
| "data pipelines / large-scale data / prototyping → production" | your platform background + the whole productionization arc |

**Framing the domain gap (genomics vs trust & safety):** the harness is domain-agnostic; genomics was a deliberate choice because ClinVar gives verifiable ground truth to develop the methodology against — and that methodology transfers directly to abuse detection, where the hard part is identical: trustworthy labels for messy, adversarial cases. The skill is the product; the domain is the testbed. (Scripted answer in `INTERVIEW_PREP.md`.)

**Leave as talking points, not builds:** RL environments for improving agent capability — understand the concept, don't build it; it's the deepest lift and not load-bearing.

---

## v2 — turn the system into an eval *platform* (post-core extension, ~Week 5+)

Ship the variant agent + harness first; it's a complete, differentiated project on its own. Then add a **second agent the same harness evaluates** — this is what turns "I built an eval for my RAG demo" into "I built general eval infrastructure for agentic systems," which is exactly what agent-eval roles want.

**The second agent: a primer-design assistant** (bridges your bench background). Natural-language request ("design Sanger primers for exon 11 of *BRCA1*") → resolve gene→coordinates (UCSC) → fetch sequence → call Primer3 → validate specificity via in-silico PCR → return.

Why it's high-leverage:
- **Objective, automatable ground truth** — Tm, GC content, amplicon size, and specificity are all *computable*. Cleaner ground truth than classification; no LLM judge needed for the core metric.
- **Proves the harness generalizes** across two structurally different agents: one fuzzy/judgment-based (classification), one deterministic/checkable (primer validity).
- **Distinctive narrative** — "I designed primers at the bench; now I build AI tooling for the bench." Also intensely visual (genome browser) → strong @digitallyiterating content.

**Pitfall to avoid:** don't wrap an LLM around Primer3 for its own sake — Primer3 is deterministic and already good. The value is the *agentic orchestration* (NL → multi-tool workflow → validation) and the *eval story*, not replacing the algorithm.

---

## How we work together (so you can articulate it, not just ship it)
The point is that *you* can explain this in an interview — so the rule is **you drive, I assist:**
1. **You state the day's goal in your own words first** — what you're building and why, in plain language and in the jargon.
2. **You attempt it** (code or design), even roughly.
3. **I refine, debug, and explain the gaps** — never just hand you a finished block to paste.
4. **End each day with a 2-min "teach-back":** you explain what you built as if to a non-technical friend (this is your content *and* your interview rep). If you can't explain it, we revisit before moving on.
5. **I never get ahead of your understanding.** If I write code, I walk you through every decision and you restate it back.

The test: at any exit gate you should be able to whiteboard that piece without notes. If you can't, it's not done — regardless of whether the code runs.
