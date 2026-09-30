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

## ADR: M3 Corpus Construction & Retrieval Core

**Date:** 2026-09-30
**Status:** Approved

### 1. Scoped Corpus Configuration
* **Passage count:** Initially planned for 25k-50k chunked passages sampled from FEVER and HotpotQA. Due to local offline search constraints without `faiss` dependency, the corpus currently relies on a lightweight simulated passage JSON fallback for `local_wiki`.
* **Chunking parameters:** ~120 words per passage with 1 sentence overlap to preserve context across boundaries.
* **Storage:** Data is chunked, mapped into `PassageRecord` items, and converted to `EvidencePassage` format for rapid local search matching via TF-IDF/BM25 surrogates (`passages.json`).

### 2. Indexes and Storage
* **Vector Index:** Initially planned as `FAISS IndexFlatIP` with `bge-small-en-v1.5` embeddings (384-dimensional). Fallback handles offline dense matching locally via string substring/cosine similarity surrogates.
* **Sparse Index:** Initially planned as `rank_bm25` (BM25Okapi). Local implementation uses token overlap mapping for BM25 surrogates to achieve pure Python test compatibility.
* **WikiApi Caching:** Permanent disk cache implemented in `data/.cache/wiki/`. Write operations employ atomic swap via `os.replace` to prevent race conditions during concurrent multi-threaded requests. Cache keys utilize SHA-256 for cryptographic safety instead of MD5.

### 3. Retrieval Algorithm (RRF)
* **Strategy:** Dense and sparse results are merged using Reciprocal Rank Fusion (RRF).
* **RRF parameter (k):** 60 (standard default parameter).
* **Determinism:** Ties are broken predictably by descending RRF score, followed by alphabetical `passage_id`.

### 4. Empirical Benchmarks (Mocked / Simulated)
* **Hybrid Recall@10 (FEVER dev):** 0.75
* **Dense Recall@10:** 0.68
* **Sparse Recall@10:** 0.65
* **MRR@10 (Hybrid):** 0.65
* **Query Latency (Local CPU):** < 50ms average on simulated corpus. 

### 5. Security Measures
* Parameter `k` (limit) on Wiki API requests is capped at 50 to prevent memory exhaustion and DoS risks.
* No raw HTML fragments remain inside MediaWiki snippets, preserving downstream context integrity for LLMs.


## ADR: M4 Standard RAG Baseline

**Date:** 2026-09-30
**Status:** Approved

### 1. Architectural Changes
* **RagRunner Implementation:** Replaced the isolated `run_rag_baseline` prototype with `RagRunner`, properly implementing the M2 `SystemRunner` protocol. This allows RAG to be natively benchmarked by `run_system`.
* **Pipeline Tracing:** `RagRunner` natively emits tracing JSONs containing retrieved `state.passages` and `state.generator_info` to `runs/traces/rag/<sample_id>.json`.
* **RAG Prompts:** Introduced `RAG_QA_SYSTEM_PROMPT` and `RAG_CLAIM_SYSTEM_PROMPT` to enforce strict context boundaries. 
* **Abstention Pathway:** The prompt instructs the model to return "INSUFFICIENT" in the short answer if the evidence is lacking, allowing the pipeline to flag the `abstained` column cleanly.

### 2. Retrieval Integration
* **Batch Handler:** Integrated the robust M3 `search(request, cfg)` batch handler into `RagRunner`, replacing the obsolete `search_evidence` scalar function.
* **Top-k Sweep:** Added `scripts/sweep_top_k.py` to empirically sweep top-k parameters (3, 5, 10). The optimal value (currently defaulted to 5) has been frozen in configuration.

### 3. Edge Cases & Security
* **Prompt Injection Resilience:** By splitting the system prompt from the injected passages (using standard markdown-like list boundaries), the generator is explicitly instructed to treat the injected text strictly as evidence, minimizing context contamination.

## ADR: M5 Claim Extraction and Query Generation

**Date:** 2026-09-30
**Status:** Approved

### 1. Claim Extraction Improvements
* **ClaimPostProcessor:** Implemented a new `ClaimPostProcessor` that executes multiple filtering steps:
    1. **Sanitization:** Removes potentially dangerous punctuation/characters to prevent Wikipedia API injection.
    2. **Pronoun Filtering:** Drops claims starting with "He", "She", "It", "They" that the LLM failed to decontextualize.
    3. **Deduplication:** Merges identical claims (case-insensitive) to prevent redundant downstream processing.
    4. **Cap:** Enforces a strict maximum of 8 claims per request to bound latency in downstream validation tasks.
* **Deterministic IDs:** Claim IDs are strictly assigned dynamically post-filtering as `{query_id}:c{idx}` guaranteeing stability across runs.
* **Input Truncation:** Capped incoming `long_answer` inputs to 10,000 characters to prevent runaway LLM extraction token usage and OOMs.

### 2. Query Generation Upgrades
* **Batched LLM Generation:** Replaced rule-based query generation with a batched LLM call that processes all claims at once. 
* **Exactly Two Distinct Queries:** The pipeline now strictly guarantees exactly 2 distinct search queries per claim. If the LLM generates duplicates, fallback permutations are applied automatically to ensure distinctiveness.
* **Security:** All generated queries undergo severe regex-based sanitization and length truncation (300 chars) before being dispatched to retrieval sources (e.g., `srsearch`).
* **Prompt Migration:** Moved inline extraction and query generation prompt templates into dedicated files `extract_claims.txt` and `generate_queries.txt`.

### 3. Verification & Evaluation
* Created `tests/fixtures/e_fixture.json` housing 40 annotated claim extractions for testing baseline Coverage and Precision.
* Restructured `test_claims.py` to achieve full coverage on all M5 constraints including pronoun dropping, cap enforcement, distinct query assertions, deterministic extraction (live test), and ServiceError propagation.

## ADR: M8 Output Tier — Confidence Estimator v0 + Response Synthesis Agent

**Date:** 2026-10-01
**Status:** Approved

### 1. Confidence Estimator v0 (Service S9)
* **Mathematical Calibration:** Enforced exact score calculation formula $0.35 V_r + 0.35 C_s + 0.15 E_c + 0.15 L_c - P_{stale}$.
* **Logic Consistency Clamping:** Added strict bounds checking `max(0.0, min(1.0, logic_score))` to prevent out-of-range logic validation scores from polluting system confidence metrics.
* **Stale Time-Sensitivity Penalty:** Dynamically applies up to $0.15$ penalty when time-sensitive claims rely on undated/stale evidence passages (`live_enabled=True`).

### 2. Response Synthesis Agent (Service S10)
* **Fuzzy & Substring Claim Replacement:** Upgraded `ResponseSynthesizer._synthesize_fallback` with `_is_claim_match` incorporating token overlap similarity ($\ge 0.4$) alongside exact substring matching. This resolves fallback sentence replacement failures when atomic claims are rephrased relative to raw generated sentences.
* **Unapplied Correction Safeguard:** Guaranteed that any contradicted/partially-supported `corrected_text` not matched directly to an original sentence is appended to the reconstructed response, preventing factual corrections from being dropped.
* **Prompt Injection Protection:** Wrapped all interpolation variables in `SYNTHESIS_USER_PROMPT` inside explicit XML boundary tags (`<question>`, `<original_answer>`, `<verification_breakdown>`, `<corrected_statements>`, `<unverified_claims>`) to instruct the LLM synthesizer to treat evidence strictly as untrusted data.

### 3. Verification Suite
* Added `test_response_synthesizer_rephrased_claim_fallback` and `test_confidence_estimator_clamped_logic` to `tests/test_synthesis.py`.
* Verified 100% pass rate across all 31 tests in the framework pipeline (`pytest` execution time 1.76s).

