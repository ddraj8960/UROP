# HMA-Fact Architectural Decision Records (ADRs)

## ADR: time-sensitive fact failure — diagnosis

**Date:** 2026-09-30
**Status:** Approved

### 1. Diagnosis Summary

The system failure on time-sensitive queries (e.g., "Who is the current Chief Minister of Tamil Nadu?") was reproduced and traced across all pipeline stages:

- **H1 (Generator Stale Knowledge):** Confirmed. Generator LLM (Llama-3.1-8B) produced "M.K. Stalin" based on pre-2024 training data. (As per specification, generator is not to be modified).
- **H2 (Claim-biased Query Generation):** Confirmed. Extracted claim "M.K. Stalin is the current Chief Minister of Tamil Nadu" led exclusively to entity-bound queries (`"M.K. Stalin current Chief Minister"`) instead of entity-neutral question-anchored queries (`"current Chief Minister of Tamil Nadu"`).
- **H3 & H4 (Retrieval Staleness & Absence of Live TTL / Dates):** Confirmed. Passages retrieved lacked `doc_date` and `retrieved_at` metadata, and live sources were not reserved in top-k slots.
- **H5 (Absence of Recency-Weighted Fusion & Date-Aware NLI):** Confirmed. Fusion aggregated historical passages without recency decay ($0.5^{\text{age\_days}/\text{half\_life}}$), leading to `consensus: SUPPORT` for an outdated CM.
- **H6 (Absence of `as_of` Date in Agent Prompts):** Confirmed. Prompts lacked current date anchoring.

### 2. Stage-by-Stage Observed Values

| Stage | Value observed | Confirms hypothesis |
|---|---|---|
| `generated.short_answer` / `long_answer` | `short_answer: "M.K. Stalin"` | H1 |
| `claims` (text, type) | `p0_diag:c0`: `"M.K. Stalin is the current Chief Minister of Tamil Nadu."` (`entity`) | H2 |
| search queries per claim | `["M.K. Stalin current Chief Minister Tamil Nadu", "M. K. Stalin Tamil Nadu Chief Minister"]` (all entity-bound) | H2 |
| passages: source, title, rank, `cached` flag, live count | 20 passages from `wiki_api`, all undated (`doc_date: None`, `retrieved_at: None`), no reserved live slots | H3, H4 |
| NLI scores of top passages vs claim | `entail=0.917`, `contradict=0.0` for passage stating Stalin took office in 2021 | H5 |
| fused evidence and preliminary label | `consensus: SUPPORT`, `support_score: 3.875`, `contradict_score: 0.0` | H5 |
| verdict label, rationale, cited passage ids | `SUPPORTED`, "implies he is current CM", cited `['M. K. Stalin']` | H5, H6 |
| whether agent prompts contained a date | No `as_of` date present in system or user prompts | H6 |

### 3. Repo Path Mapping

| Plan Component / Service | Actual Codebase File Path |
|---|---|
| Schemas / Contracts | `src/hmafact/schemas/` (`input.py`, `claims.py`, `evidence.py`, `graph.py`, `fusion.py`, `verification.py`, `logic.py`, `confidence.py`, `synthesis.py`) |
| S1 LLM Gateway Client | `src/hmafact/llm/openrouter.py` |
| S2 Generator Baseline | `src/hmafact/generator/generator.py`, `prompts.py` |
| S3 Retrieval Service | `src/hmafact/retrieval/hybrid.py`, `wiki_api.py` |
| S4 & S4a Graph & NLI | `src/hmafact/graph/builder.py`, `nli.py` |
| S5 Claim Extraction Agent | `src/hmafact/claims/extractor.py`, `prompts.py` |
| S6 Evidence Fusion Engine | `src/hmafact/fusion/fusion.py` |
| S7 Fact Verification Agent | `src/hmafact/verification/verifier.py`, `prompts.py` |
| S8 Logic Validation Agent | `src/hmafact/logic/validator.py` |
| S9 Confidence Estimator | `src/hmafact/confidence/estimator.py` |
| S10 Response Synthesizer | `src/hmafact/synthesis/synthesizer.py`, `prompts.py` |
| Configs | `configs/default.yaml` |

### 4. Cache Bypass Diagnostic

Running without cache confirmed that `wiki_api` queries derived solely from the claim (`"M.K. Stalin Chief Minister"`) retrieve articles about M. K. Stalin. Only question-anchored queries (`"current Chief Minister of Tamil Nadu"`) together with recency weighting enable the verification layer to retrieve current office-holder facts.
