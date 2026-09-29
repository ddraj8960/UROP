# HMA-Fact — Master Build Plan (Single Source of Truth)

**Project:** Hierarchical Multi-Agent Fact Verification using Dynamic Retrieval Graphs for Hallucination Reduction in LLMs
**Team:** 2 undergraduates, working sequentially on one module at a time
**Compute:** Cloud only (Colab Pro + NVIDIA Build API; OpenRouter as fallback)
**Document status:** v1.0, day zero. Every later per-module build spec must be derived from this file. If a later spec needs to change a decision here, change this file first (bump the version in §0.3) and then write the spec.

---

## 0. How to Use This Document

### 0.1 Conventions that every module spec must obey

These are global rules. Later per-module specs inherit them and must not override them.

1. **Library-first, UI-last.** All logic lives in the Python package `hmafact`. The API and the dashboard only *call* the package. No logic goes into Streamlit or FastAPI files.
2. **Pydantic v2 models are the contracts.** Every module takes and returns objects defined in `hmafact/schemas/`. Modules never pass raw dicts across a module boundary.
3. **Every LLM call goes through one client** (`hmafact/llm/client.py`), which handles provider switching, retries, JSON validation, caching, and token and latency accounting. No module imports `openai` or `requests` directly.
4. **Everything is cached and deterministic.** Temperature is 0, seeds are fixed, and there is an on-disk cache keyed by `(provider, model, messages, params)`. Re-running an experiment costs nothing unless the prompt changes.
5. **Every tunable lives in YAML config** (`configs/`). This includes model names, `k` values, thresholds, and ablation switches. Code reads settings from the config object and never hard-codes them.
6. **Every pipeline run emits a `PipelineTrace`** (defined in §3.3). The dashboard, the evaluation code, and debugging all read traces. This is the most important design decision in the plan.
7. **Each module has pure functions with signature `run(state_in, config) -> state_out`.** LangGraph only wires these functions together. Each one can therefore be unit-tested without LangGraph.
8. **Tooling:** Python 3.11, `uv` for environments, `pytest`, `ruff`, type hints everywhere, and `.env` for secrets (never committed).

### 0.2 Canonical verification labels (fixed; used everywhere)

`SUPPORTED`, `CONTRADICTED`, `PARTIALLY_SUPPORTED`, `INSUFFICIENT_EVIDENCE`, `CONFLICTING_EVIDENCE`

### 0.3 Version log
- v1.0: initial plan.
- v1.2: additive contract fields `UserQuery.as_of`, `Claim.time_sensitive`, `Claim.temporal_scope`, `EvidencePassage.retrieved_at`, `EvidencePassage.doc_date`; new config block `live`; new behaviour behind `live.enabled` (default false): time-sensitivity detection, question-anchored queries, reserved live slots, API-cache TTL, recency-weighted fusion, date-aware verification, temporal logic check, "as of" synthesis. No change to benchmark behaviour.


---

## 1. Literature Survey Plan

### 1.1 Scope and timing

- **Target:** 22–25 papers read and entered into the survey matrix, then 15–20 cited in depth in D2. The brief asks for 15–20; reading a few extra gives room to cut.
- **Time box:** Weeks 1–2 are survey-heavy. Engineering Module M0 starts in the second half of Week 2, because it needs no research input. The survey is *closed* at the end of Week 3 with a written gap statement. Papers found after that go into an "additional reading" list only.
- **Date window:** 2021–2026 for methods. Older papers only for the three benchmark papers and the foundational RAG and NLI papers.

### 1.2 Search strategy

- **Sources, in order:** Semantic Scholar (best citation graph), ACL Anthology (EMNLP/ACL/NAACL), arXiv (cs.CL, cs.AI), Google Scholar (coverage check only).
- **Search strings** (run each; save the top 20 hits per string):
  - `"hallucination detection" LLM claim-level`
  - `"atomic facts" OR "atomic claims" factuality evaluation`
  - `"fact verification" retrieval LLM agent`
  - `"multi-agent" "fact checking" OR "fact verification"`
  - `"retrieval-augmented generation" verification OR corrective OR self-reflection`
  - `"evidence graph" OR "knowledge graph" fact verification`
  - `"confidence calibration" LLM abstention OR "selective prediction"`
- **Snowballing:** For each of the anchor papers in §1.4, scan the backward references and the "cited by" list on Semantic Scholar, filtered to highly cited or recent papers. This finds most of the relevant work faster than keyword search.
- **Inclusion rule:** A paper is included only if it (a) proposes a method in hallucination detection, mitigation, or verification, (b) introduces or seriously uses one of our three benchmarks, or (c) is a survey. Every paper must map to at least one of our nine components.

### 1.3 Splitting the work between two people

Because you work sequentially, split *reading* but not *synthesis*:

| Who | Theme | Covers components |
|---|---|---|
| Both, together (Week 1) | The anchor papers in §1.4 marked ★, plus one survey | Common vocabulary |
| Person A | Factuality evaluation and detection: claim decomposition, self-consistency, NLI-based verification, calibration and abstention | 9.3, 9.7, 9.9, eval |
| Person B | Retrieval-side mitigation: RAG variants, corrective and iterative retrieval, graph-based RAG, multi-agent pipelines and debate | 9.4, 9.5, 9.6, 9.8, orchestration |

Hold a 45-minute joint session twice a week. Each person presents their new matrix rows, and together you update the gap table (below).

### 1.4 Tracking findings so that a gap emerges

- **Reference manager:** a shared Zotero group library (free) with tags `A-eval`, `B-retrieval`, `anchor`, `benchmark`.
- **Survey matrix:** a shared Google Sheet with one row per paper and these columns:

`Paper | Year | Venue | Core idea (1 line) | Claim-level? | Retrieval? | Multi-source fusion? | Explicit graph structure? | Logic / cross-claim consistency? | Calibrated confidence? | Abstention / correction? | Multi-agent / hierarchical? | Benchmarks | Generator model | Reported gain | Limitation (their own words, paraphrased) | Relevance to our module(s)`

- **How the gap is found:** The Y/N capability columns are the gap detector. At the end of Week 3, filter the sheet. You should find that no single system combines *claim-level* + *explicit per-query evidence graph* + *multi-source fusion* + *cross-claim logic check* + *calibrated confidence with abstention*. That empty combination of columns *is* your research gap. It matches the brief's contribution statement, and the sheet is the evidence for it. Put a condensed version of this table in D2 as "Table: Capability comparison of prior work."
- **Deliverable:** D2 draft of about 3,000 words, containing the thematic review, the capability table, a gap statement, and the mapping of research questions RQ1–RQ5 to prior work.

### 1.5 Anchor papers and systems (starting set)

Check each citation's details in Semantic Scholar before citing. ★ = read together in Week 1.

1. ★ **Huang et al., 2023**, "A Survey on Hallucination in Large Language Models." Taxonomy and background framing.
2. ★ **Min et al., 2023, FActScore** (EMNLP). Decomposes text into atomic facts and verifies each against Wikipedia. This is the direct ancestor of your claim extraction and claim-level scoring.
3. ★ **Wei et al., 2024, "Long-form factuality in LLMs" (SAFE / LongFact).** An LLM agent that decomposes claims and verifies each with search. The closest "agentic verification" prior work.
4. ★ **Dhuliawala et al., 2023, Chain-of-Verification (CoVe).** The model plans and answers its own verification questions. A single-agent baseline for your hierarchical design.
5. **Manakul et al., 2023, SelfCheckGPT.** Zero-resource detection via sampling consistency. A useful contrast, since it needs no retrieval.
6. **Gao et al., 2023, RARR.** Retrieve-then-revise attribution and editing. The ancestor of your correction step in synthesis.
7. **Asai et al., 2023, Self-RAG**, and **Yan et al., 2024, Corrective RAG (CRAG).** Retrieval that reflects on or corrects itself. The main "smarter RAG" competitors.
8. **Chern et al., 2023, FacTool.** A tool-augmented, multi-domain factuality detection framework.
9. **Pan et al., 2023, ProgramFC.** Decomposes claims into reasoning programs for multi-hop fact-checking. Relevant to logic validation and multi-hop HotpotQA.
10. **Edge et al., 2024, GraphRAG (Microsoft).** Graph-structured retrieval. Relevant for positioning your evidence graph; note that theirs is corpus-level while yours is per-query.
11. **Du et al., 2023, "Improving Factuality and Reasoning through Multiagent Debate."** The multi-agent factuality baseline idea.
12. **Benchmark papers (must cite):** Thorne et al., 2018 (FEVER); Yang et al., 2018 (HotpotQA); Lin et al., 2022 (TruthfulQA).
13. **Calibration:** Guo et al., 2017, "On Calibration of Modern Neural Networks" (ECE and temperature scaling); Kadavath et al., 2022, "Language Models (Mostly) Know What They Know."

---

## 2. Finalized Technology Stack

These are decisions, not a menu. Model identifiers must go in `configs/default.yaml` because provider catalogs change. **Verify the exact model IDs on build.nvidia.com and openrouter.ai when you build M0.**

| Layer | Decision | Why | Cost / GPU notes |
|---|---|---|---|
| **Generator LLM (9.2)**, the model being verified | **Llama-3.1-8B-Instruct** via NVIDIA Build (OpenAI-compatible endpoint). Same model on OpenRouter as fallback. | Small enough to hallucinate measurably, so there is room to show a reduction; open-weight, so it is reproducible. A frontier generator would leave little headroom to measure improvement. | API only, no GPU. NVIDIA Build gives free trial credits but is rate-limited. The 8B model is cheap on OpenRouter. |
| **Agent LLM** (claim extraction, query generation, verification adjudication, logic check, synthesis) | **Llama-3.3-70B-Instruct** via NVIDIA Build, with OpenRouter as fallback. | Much better instruction following and JSON reliability than 8B, which matters for structured agent outputs. Keeping one model family simplifies the story. | About 8–12 calls per query. Caching makes ablations nearly free. See §8.5 for a budget estimate. |
| **Evaluation judge LLM** | A **different model family** from the agents (e.g. a Qwen or Mistral large instruct model via OpenRouter). Fix it once and never change it. | Avoids self-preference bias, which examiners and reviewers will ask about. | Used only in evaluation. |
| **NLI model** (claim–evidence entailment scoring) | **`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`** (Hugging Face) as a cross-encoder. | Fast, strong, and deterministic; gives entail/neutral/contradict probabilities that feed graph edges and confidence features. Reduces LLM calls. | Runs on CPU for the demo (tens of pairs per second in batches). Use Colab T4 for batch evaluation. **Disclose** that it was trained partly on FEVER *train* data; you evaluate on FEVER *dev*. |
| **Orchestration** | **LangGraph** | An explicit state graph with conditional edges directly implements "hierarchical plus re-retrieval loop." It is deterministic, well documented, and Claude Code handles it well. CrewAI's role-play autonomy makes runs less reproducible, which hurts evaluation. | CPU. |
| **Embeddings** | **`BAAI/bge-small-en-v1.5`** (384 dimensions, sentence-transformers) | Strong retrieval quality for its size; embedding about 300k passages takes well under an hour on a Colab T4. | One-time GPU cost for the index; queries embed on CPU. |
| **Lexical retrieval** | **`bm25s`** (pure Python, fast BM25) | Hybrid BM25 plus dense retrieval beats either alone on entity-heavy FEVER and HotpotQA. It is also used to build the scoped corpus (§4, M3). | CPU. |
| **Vector store** | **FAISS** (`IndexFlatIP` over normalized vectors; switch to HNSW only if the corpus exceeds about 1M passages) | Simplest, fastest, no server, saves to a single file on Google Drive. Chroma adds a service and metadata layer you don't need, since metadata lives in a Parquet sidecar. | Index for 300k passages at 384d is about 460 MB. |
| **Knowledge sources** (2–3, per brief) | **S1:** local Wikipedia corpus (FEVER and HotpotQA Wikipedia dumps, scoped). **S2:** live Wikipedia API search (cached). **S3 (from M11):** Wikidata SPARQL for dates, numbers, and entity facts. | Three sources that genuinely differ in kind: static text, current text, and structured data. This gives the fusion agent real heterogeneity and real conflicts to handle. | Free, no GPU. Cache every API response. |
| **Evidence graph** | **NetworkX `MultiDiGraph`**, serialized with `node_link_data` to JSON | Required by the brief; trivial to serialize into traces; easy to visualize. Neo4j is out of scope because it adds a server for no functional gain at this scale. | CPU. |
| **Data handling** | pandas + pyarrow (Parquet); spaCy `en_core_web_sm` (entities, dates); `dateparser` | Standard and well supported. | CPU. |
| **Backend** | **FastAPI** thin wrapper (`/verify`, `/traces/{id}`, `/runs`) | Standard; allows the dashboard to call the pipeline asynchronously; gives you a demoable API. | CPU. |
| **Dashboard** | **Streamlit** (multipage) + **Plotly** (charts) + **pyvis** (interactive graph embedded via `st.components.v1.html`) | Fastest route to a polished demo; Claude Code is very reliable with it. | CPU. |
| **Storage** | Traces as JSON files plus a **SQLite** index; evaluation predictions as JSONL; metrics as JSON; aggregated results as Parquet | Only files, so no database server; easy to copy to Drive and to replay offline. | none |
| **LLM cache** | `diskcache` (SQLite-backed) | One line of code; persistent across Colab sessions if stored on Drive. | none |
| **Evaluation** | scikit-learn, scipy (bootstrap, McNemar via statsmodels), custom metrics | Standard. | CPU. |

**Where the faculty-suggested tools fit:**
- **NVIDIA Build:** the primary LLM provider for the generator and agents. Its endpoint is OpenAI-compatible, so the client uses the `openai` SDK with `base_url` changed.
- **OpenRouter:** the fallback provider (same client, different `base_url`), and the home of the cross-family evaluation judge. Switching providers is one line in the config.
- **Kilo Code:** an open-source coding agent for VS Code. Since Claude Code is your primary builder, use Kilo Code only for small edits or when you want to try an OpenRouter-hosted model on refactors. Don't split one module's build across both tools, because it creates inconsistent code style.

**Where GPU hours actually go** (budget for about 6–10 Colab GPU hours in total): building embedding indexes (about 1–2 h), batched NLI over evaluation sets (about 2–4 h across all runs, since the NLI output is cached), and spare time for re-indexing. Everything else runs on a laptop CPU plus APIs.

---

## 3. Repository and System Structure

### 3.1 Folder layout

```
hma-fact/
├── pyproject.toml              # uv-managed; package name: hmafact
├── .env.example                # NVIDIA_API_KEY, OPENROUTER_API_KEY, HMA_DATA_DIR
├── configs/
│   ├── default.yaml            # models, k values, thresholds, paths
│   └── experiments/            # vanilla.yaml, rag.yaml, hmafact.yaml, ablation_*.yaml
├── src/hmafact/
│   ├── config.py               # pydantic-settings loader, config hash
│   ├── schemas/                # ALL interface contracts (§3.3)  ← M0
│   ├── llm/                    # client.py, cache.py, prompts/*.jinja  ← M0
│   ├── tracing/                # trace writer, SQLite index, timers  ← M0
│   ├── data/                   # benchmark loaders, fixed sample splits  ← M1
│   ├── generator/              # 9.2 Initial LLM Response Generator  ← M2
│   ├── corpus/                 # corpus scoping + index build scripts  ← M3
│   ├── retrieval/              # 9.4 retrievers (local hybrid, wiki API, wikidata)  ← M3/M6/M11
│   ├── baselines/              # vanilla.py, rag.py  ← M2/M4
│   ├── claims/                 # 9.3 Claim Extraction Agent  ← M5
│   ├── graph/                  # 9.5 Dynamic Evidence Retrieval Graph  ← M6
│   ├── fusion/                 # 9.6 Evidence Fusion Agent  ← stub M7, real M11
│   ├── verification/           # 9.7 Fact Verification Agent (NLI + LLM)  ← M7
│   ├── logic/                  # 9.8 Logic Validation Agent  ← M12
│   ├── confidence/             # 9.9 Confidence Estimation Agent  ← M8 v0, M13 calibrated
│   ├── synthesis/              # 9.10 Response Synthesis Agent  ← M8
│   ├── orchestration/          # LangGraph graph + ablation switches  ← M9
│   ├── eval/                   # metrics, judge, runner, stats, ablations  ← M2→M14
│   └── api/                    # FastAPI app  ← M15
├── dashboard/                  # 9.1 User Interface (Streamlit)  ← M10/M15
│   ├── app.py
│   └── pages/ 1_Live_Demo.py, 2_Results.py
├── scripts/                    # CLI entry points: build_index, run_eval, calibrate
├── notebooks/colab/            # thin Colab notebooks that call scripts (GPU jobs only)
├── tests/                      # mirrors src/; fixtures/ holds tiny gold sets
├── data/  indexes/  runs/      # gitignored; symlinked to Google Drive on Colab
└── docs/                       # architecture.md, decisions.md (ADR log), this plan
```

### 3.2 Runtime data flow

```
UserQuery
  → [Generator] → GeneratedResponse
  → [ClaimExtractor] → list[Claim]
  → [Retriever: per-claim queries × sources] → writes Query/Passage/Document/Source nodes
  → [EvidenceGraph] (NLI scores claim↔passage edges; entity links; cross-claim sharing)
  → [Fusion] → dict[claim_id, FusedEvidence]
  → [Verifier] → dict[claim_id, ClaimVerdict]
  → [LogicValidator] → list[LogicIssue]  ──(flagged claims, iteration < max)──→ back to Retriever
  → [Confidence] → dict[claim_id, ConfidenceScore] + response-level score
  → [Synthesis] → FinalResponse
  → PipelineTrace written to runs/
```

### 3.3 Interface contracts (`src/hmafact/schemas/`)

These are the shared contracts that make the modules independently buildable. Field names are fixed. Modules may add *optional* fields, but must not rename or remove existing ones.

```python
# common.py
Label = Literal["SUPPORTED","CONTRADICTED","PARTIALLY_SUPPORTED",
                "INSUFFICIENT_EVIDENCE","CONFLICTING_EVIDENCE"]
SourceName = Literal["local_wiki","wiki_api","wikidata"]

class LLMCallRecord(BaseModel):          # appended by llm client to trace
    stage: str; provider: str; model: str
    prompt_tokens: int; completion_tokens: int
    latency_ms: float; cached: bool

# input.py
class UserQuery(BaseModel):
    query_id: str                        # uuid or benchmark sample_id
    text: str
    mode: Literal["qa","claim"] = "qa"   # "claim" = FEVER: skip generator/extractor
    dataset: str | None = None           # "truthfulqa"|"fever"|"hotpotqa"|None(live)

class BenchmarkSample(BaseModel):
    sample_id: str; dataset: str; split: Literal["dev","test"]
    input_text: str                      # question or claim
    gold_answer: str | None              # HotpotQA answer / TruthfulQA best answer
    gold_correct_answers: list[str] = [] # TruthfulQA
    gold_incorrect_answers: list[str] = []
    gold_label: Literal["SUPPORTS","REFUTES","NOT ENOUGH INFO"] | None = None  # FEVER
    gold_evidence: list[tuple[str,int]] = []   # (wiki_title, sentence_idx) / supporting facts
    meta: dict = {}                      # e.g. hotpot type bridge/comparison

# generation.py
class GeneratedResponse(BaseModel):
    query_id: str
    short_answer: str                    # concise answer (EM scoring, HotpotQA)
    long_answer: str                     # explanatory text claims are extracted from
    model: str

# claims.py
class Claim(BaseModel):
    claim_id: str                        # f"{query_id}:c{n}"
    text: str                            # decontextualized, standalone atomic claim
    source_span: str | None              # span of long_answer it came from
    entities: list[str] = []
    claim_type: Literal["entity","temporal","numeric","relational","other"] = "other"

# evidence.py
class EvidencePassage(BaseModel):
    passage_id: str                      # stable: f"{source}:{doc_id}:{chunk_idx}"
    source: SourceName
    doc_id: str; title: str; text: str
    url: str | None = None
    retrieval_score: float               # fused hybrid score, normalized 0-1
    rank: int
    retrieved_for: list[str]             # claim_ids (graph dedups across claims)
    search_query: str

class NLIScores(BaseModel):
    entail: float; neutral: float; contradict: float   # sum to 1

# graph.py  (graph itself is networkx; this is its serialized trace form)
class GraphSnapshot(BaseModel):
    node_link: dict                      # nx.node_link_data(G)
    iteration: int
# Node types (attr "ntype"): QUERY, CLAIM, SEARCH_QUERY, PASSAGE, DOCUMENT, SOURCE, ENTITY
# Edge types (attr "etype"): HAS_CLAIM, GENERATED_QUERY, RETRIEVED, PART_OF,
#   FROM_SOURCE, MENTIONS, SUPPORTS, CONTRADICTS, RELATED_TO, CONFLICTS_WITH, DERIVED_FROM
# Evidence edges (PASSAGE→CLAIM) carry: nli: NLIScores, relevance: float

# fusion.py
class FusedEvidence(BaseModel):
    claim_id: str
    selected_passage_ids: list[str]      # ordered, ≤ config.fusion.max_evidence
    support_mass: float                  # credibility- & relevance-weighted entail
    contradict_mass: float
    n_sources_supporting: int
    n_sources_contradicting: int
    agreement: float                     # 0..1
    conflict_detected: bool
    prelim_label: Label                  # rule-based suggestion

# verification.py
class ClaimVerdict(BaseModel):
    claim_id: str
    label: Label
    rationale: str
    cited_passage_ids: list[str]
    corrected_fact: str | None = None    # when CONTRADICTED/PARTIAL and evidence gives truth
    verbal_confidence: float | None = None   # LLM self-report, feature only

# logic.py
class LogicIssue(BaseModel):
    issue_type: Literal["temporal","numeric","entity","cross_claim","answer_mismatch"]
    claim_ids: list[str]
    description: str
    severity: Literal["low","high"]
    trigger_reretrieval: bool

# confidence.py
class ConfidenceScore(BaseModel):
    claim_id: str
    score: float                         # calibrated in M13; heuristic before
    features: dict[str, float]           # logged for calibration training
    band: Literal["high","verified","qualify","abstain"]  # 0.90/0.75/0.50 cutoffs from config

# output.py
class FinalResponse(BaseModel):
    query_id: str
    short_answer: str | None             # None if abstained
    answer_text: str
    abstained: bool
    response_confidence: float
    fever_label: Literal["SUPPORTS","REFUTES","NOT ENOUGH INFO"] | None = None
    citations: dict[str, list[str]]      # sentence_idx -> passage_ids
    kept_claim_ids: list[str]; corrected_claim_ids: list[str]; dropped_claim_ids: list[str]

# state.py — LangGraph state AND the trace payload
class PipelineState(BaseModel):
    query: UserQuery
    config_hash: str
    generated: GeneratedResponse | None = None
    claims: list[Claim] = []
    passages: dict[str, EvidencePassage] = {}
    graph: GraphSnapshot | None = None
    fused: dict[str, FusedEvidence] = {}
    verdicts: dict[str, ClaimVerdict] = {}
    logic_issues: list[LogicIssue] = []
    confidences: dict[str, ConfidenceScore] = {}
    final: FinalResponse | None = None
    iteration: int = 0
    stage_timings_ms: dict[str, float] = {}
    llm_calls: list[LLMCallRecord] = []
    errors: list[str] = []

class PipelineTrace(BaseModel):          # = serialized final PipelineState + metadata
    run_id: str; system: Literal["vanilla","rag","hmafact"] | str
    state: PipelineState
    total_latency_ms: float; git_commit: str; created_at: datetime
```

The baselines (vanilla and RAG) also produce a `PipelineTrace`, with only `generated`, `passages` (RAG only), and `final` filled in. This is what lets one evaluator and one dashboard handle all three systems.

---

## 4. Module-by-Module Build Plan

Work down this list top to bottom. **Do not start module *n*+1 until module *n* meets its "Done when" criteria.** Each module lists a *week target* against the 14-week plan in the brief.

Module template: **Maps to** (brief component / objective / deliverable) · **Depends on** · **Input → Output** · **Key decisions** · **Test in isolation** · **Done when**

---

### M0 — Foundations (Week 2–3)
- **Maps to:** D3, D13 groundwork.
- **Depends on:** nothing.
- **Builds:** repository skeleton (§3.1); `config.py` (YAML plus `.env`, with a config hash); all schemas from §3.3; `llm/client.py`; `llm/cache.py`; `tracing/`.
- **LLM client contract:** `complete(stage: str, messages, *, model_role: Literal["generator","agent","judge"], json_schema: type[BaseModel] | None, **params) -> (text | BaseModel, LLMCallRecord)`. Behavior: OpenAI SDK pointed at the configured provider `base_url`; exponential-backoff retry on 429 and 5xx errors; automatic failover to the secondary provider after N failures; when `json_schema` is set, request JSON mode, validate with Pydantic, and on failure re-prompt once with the validation error before raising; cache lookup comes first.
- **Tracing contract:** `TraceWriter.write(trace) -> path` saves to `runs/{run_id}/traces/{query_id}.json` and adds a row to `runs/index.sqlite` (`run_id, query_id, system, dataset, latency, abstained, created_at`). Provide a `stage_timer(state, "claims")` context manager.
- **Test:** round-trip test for every schema (serialize to JSON and back); client test with a mocked HTTP layer (retry, failover, JSON repair, cache hit); one real "hello" call per provider (marked `@pytest.mark.live`).
- **Done when:** `uv run pytest` passes; a real call to NVIDIA Build and to OpenRouter succeeds through the client; a second identical call returns `cached=True`; a dummy trace is written and indexed.

### M1 — Benchmark Data and Fixed Splits (Week 3)
- **Maps to:** O9 groundwork, D12.
- **Depends on:** M0.
- **Input → Output:** Hugging Face datasets → `data/samples/{dataset}_{split}.jsonl` of `BenchmarkSample`.
- **Key decisions:**
  - TruthfulQA: the `generation` configuration (817 questions). Sample 100 for dev and 300 for test, stratified by category.
  - FEVER: the labelled dev set. Sample 100 for dev and 300 for test, balanced 1/3 each across SUPPORTS / REFUTES / NEI.
  - HotpotQA: the **distractor** setting, dev set. Sample 100 for dev and 300 for test, stratified by type (bridge or comparison) and keeping the hard level.
  - Fixed seed 42. The split files are committed to git so the splits are frozen forever. **Dev** is used for all prompt tuning, threshold tuning, and calibration. **Test** is touched only by final runs (M14).
- **Test:** schema validation of every row; stratification counts checked; no ID overlap between dev and test.
- **Done when:** six JSONL files exist with the correct counts, and a `data/README.md` records their sources, versions, and seed.

### M2 — Generator + Vanilla LLM Baseline + Evaluation Harness v1 (Week 4)
- **Maps to:** 9.2, D4, O9 (Baseline 1).
- **Depends on:** M0, M1.
- **Input → Output:** `UserQuery` → `GeneratedResponse`; the vanilla baseline wraps it as `FinalResponse` (no abstention) plus a trace.
- **Key decisions:**
  - The generator prompt returns JSON `{short_answer, long_answer}`, where `long_answer` is 2–5 sentences of justification. Keeping it short keeps the number of claims manageable.
  - In `claim` mode (FEVER), the vanilla system is asked to classify the claim as SUPPORTS / REFUTES / NOT ENOUGH INFO with no evidence.
  - **Evaluation harness v1** (`eval/`): `run_system(system, split, dataset) -> runs/{run_id}/predictions.jsonl`, plus scorers: HotpotQA EM/F1 (official normalization); FEVER accuracy and macro-F1; TruthfulQA **LLM-judge** (judge model, prompt shows the question plus correct and incorrect reference answers, output `{truthful: bool, informative: bool}`). Also an answer-level judge for HotpotQA semantic correctness (EM is too strict for LLM wording). All metrics go to `metrics.json`.
  - **Hallucination rate (fixed definition for the whole project):** among *non-abstained* answers, the fraction that is judged incorrect (TruthfulQA: untruthful; HotpotQA: judge says incorrect; FEVER: wrong label, where predicting NEI counts as abstention). Always report the abstention rate alongside it.
- **Test:** scorer unit tests against hand-computed examples; run the vanilla system on 20 dev samples per dataset.
- **Done when:** the vanilla system has run on all three **dev** splits, `metrics.json` exists, and you have **hand-checked 50 judge decisions** and recorded agreement (the target is ≥85%; if lower, fix the judge prompt now).

### M3 — Corpus Construction and Retrieval Core (Week 5)
- **Maps to:** 9.4 (partial), O2, D7 groundwork.
- **Depends on:** M1.
- **Input → Output:** raw dumps → `indexes/local_wiki/{faiss.index, passages.parquet, bm25/}`. Runtime: `LocalHybridRetriever.search(query: str, k: int) -> list[EvidencePassage]`.
- **Key decisions (the scoped-corpus strategy, which avoids embedding 5M+ pages):**
  1. **HotpotQA:** all context paragraphs from every sampled dev and test question (gold plus distractors). This matches the benchmark's intended evidence pool.
  2. **FEVER:** download the FEVER `wiki-pages` dump; build a BM25 index over *page intros* (CPU, with bm25s). For each sampled claim, take the pages in its gold evidence plus the top 30 BM25 pages. Add 20k random pages as background distractors so the task isn't artificially easy.
  3. **TruthfulQA:** fetch the Wikipedia pages behind each question's `source` URL (where it is a Wikipedia URL) plus the top 10 Wikipedia API search results per question, once, and cache them to disk.
  4. Chunk text into passages of about 120 words with a 1-sentence overlap, keeping `title`. Expect roughly 150k–400k passages.
  5. Embed with bge-small on Colab (with bge's query instruction prefix for queries). FAISS flat inner-product index. BM25 over the same passages. **Hybrid score:** reciprocal rank fusion (RRF, k=60) of the dense top-50 and BM25 top-50, then take the top `k` (default 5), with scores normalized to 0–1.
  - Record corpus statistics in `indexes/local_wiki/manifest.json`.
  - **Limitation to disclose honestly in the report:** the corpus is scoped from the sampled items. This is standard practice for compute-limited work; the distractors keep it non-trivial.
- **Test:** retrieval metrics on dev: FEVER gold-page Recall@5/10, HotpotQA supporting-paragraph Recall@5/10, MRR. Latency per query below 300 ms on CPU.
- **Done when:** the index is built and saved to Drive; it reloads on a laptop CPU; dev Recall@10 is ≥0.7 on HotpotQA and FEVER (if lower, increase the BM25 weight or `k` before moving on). Numbers are logged in `docs/decisions.md`.

### M4 — Standard RAG Baseline (Week 6)
- **Maps to:** D5, O9 (Baseline 2).
- **Depends on:** M2, M3.
- **Input → Output:** `UserQuery` → retrieve top-5 on the *original query* → generator with context → `FinalResponse` plus a trace (with `passages`).
- **Key decisions:** The same generator model and the same output JSON as vanilla; the only change is the context block. There is no abstention instruction beyond "say you don't know if the context doesn't contain the answer" (the conventional RAG prompt). In FEVER mode, it classifies the claim given the evidence.
- **Test:** run on all dev splits.
- **Done when:** a **dev baseline table** (Vanilla vs RAG, all three datasets) is generated by `scripts/report.py`. You now have Baselines 1 and 2, and this is the first thing to show your guide.

### M5 — Claim Extraction Agent (Week 7)
- **Maps to:** 9.3, O1, D6.
- **Depends on:** M0, M2.
- **Input → Output:** `GeneratedResponse` → `list[Claim]`.
- **Key decisions:**
  - Agent LLM with a few-shot prompt (5 examples written in the style of FActScore's atomic facts). Each claim must be **decontextualized**: pronouns resolved and the subject named, so that "He was born in 1956" becomes "Tom Hanks was born in 1956".
  - The agent also returns `entities` and `claim_type`. spaCy post-processing fills in missing entities.
  - Cap of 8 claims per answer (config). Deduplicate near-identical claims (embedding cosine above 0.92).
  - Claims that express opinions or are not verifiable are dropped and logged.
  - The question itself is passed as context so that claims implied by the answer (e.g. "the answer is X") include the question's subject.
  - In `claim` mode (FEVER), this module passes the input claim through as a single `Claim` without calling the LLM.
- **Test:** hand-build `tests/fixtures/claims_gold.jsonl` with **40 generated answers** (sampled from M2 outputs on dev) and annotated gold atomic claims. Metrics: claim-count error, and semantic coverage and precision via embedding matching at cosine ≥0.8 (plus spot-checking).
- **Done when:** coverage ≥0.85 and precision ≥0.85 on the fixture set; no pronoun-only claims appear in a manual check of 20 outputs.

### M6 — Dynamic Evidence Graph + Claim-Level Multi-Source Retrieval (Weeks 8–9)
- **Maps to:** 9.4, 9.5, O2, O3, D7, D8.
- **Depends on:** M3, M5.
- **Input → Output:** `UserQuery` + `list[Claim]` → `EvidenceGraph` object (wrapping `nx.MultiDiGraph`) + `dict[passage_id, EvidencePassage]` + `GraphSnapshot`.
- **Key decisions:**
  - **Per-claim query generation:** the agent LLM writes 2 search queries per claim (one entity-focused, one paraphrase of the claim). One batched call covers all claims in the answer.
  - **Sources in this module:** S1 `local_wiki` (hybrid) and S2 `wiki_api` (MediaWiki search plus page extracts; cached; chunked the same way as S1). S3 Wikidata comes in M11.
  - **Graph construction (`graph/builder.py`), in this order:**
    1. Add QUERY, CLAIM, and SEARCH_QUERY nodes and their edges.
    2. Add PASSAGE nodes (deduplicated by `passage_id` across claims), with DOCUMENT and SOURCE nodes linked by `PART_OF` and `FROM_SOURCE`.
    3. Add ENTITY nodes from spaCy over claims and passages, with `MENTIONS` edges.
    4. **Cross-claim evidence sharing:** every passage is scored against *every* claim that shares at least one entity with it, not just the claim it was retrieved for.
    5. **Entity-hop expansion (one hop, config switch):** entities that appear in a claim's top evidence but not in the claim itself trigger one extra retrieval ("bridge" entities). This targets multi-hop HotpotQA.
  - **NLI scoring lives in the graph layer** (`graph/nli.py`): for each candidate (passage, claim) pair, run the DeBERTa NLI model (premise = passage, hypothesis = claim). Add edges `SUPPORTS` (entail ≥ τ_e, default 0.6), `CONTRADICTS` (contradict ≥ τ_c, default 0.6), or otherwise `RELATED_TO`, each carrying `nli` and `relevance`.
  - **API:** `EvidenceGraph.evidence_for(claim_id) -> list[(EvidencePassage, NLIScores, etype)]`, `.snapshot(iteration)`, `.add_retrieval(...)` (idempotent, so the M12 re-retrieval loop can grow the graph: this is the "dynamic" part).
  - **Ablation switch** `graph.enabled=false`: fall back to flat per-claim lists (no cross-claim sharing and no entity hop, but NLI still runs). This is essential for RQ2.
- **Test:** fixture queries with known evidence; assertions on node and edge counts and types; idempotency (adding the same retrieval twice changes nothing); evidence recall per claim on 30 FEVER dev claims (gold page present in `evidence_for`); snapshot round-trip; render a pyvis HTML for 3 examples and inspect them by eye.
- **Done when:** claim-level evidence Recall@10 on FEVER dev is at least the M3 query-level figure; graph-on vs graph-off shows higher evidence coverage on HotpotQA bridge questions (even a small gain counts; log it); snapshots serialize into traces.

### M7 — Fact Verification Agent + Fusion Stub (Week 10)
- **Maps to:** 9.6 (stub), 9.7, O5, D9 groundwork.
- **Depends on:** M6.
- **Input → Output:** `EvidenceGraph` + `claims` → `dict[claim_id, FusedEvidence]` (stub) → `dict[claim_id, ClaimVerdict]`.
- **Key decisions:**
  - **Fusion stub:** select the top `max_evidence=5` passages by `relevance × max(entail, contradict)`; compute support and contradiction mass *unweighted*; `prelim_label` comes from simple rules. The real version comes in M11, with the **same output schema**.
  - **Verifier:** the agent LLM receives the claim, the selected passages (with IDs and sources), and the NLI summary. It returns a `ClaimVerdict` via JSON schema, with definitions of the five labels included verbatim in the prompt. It must cite passage IDs; a verdict of SUPPORTED or CONTRADICTED that cites nothing is downgraded to INSUFFICIENT_EVIDENCE by a post-check.
  - For CONTRADICTED or PARTIALLY_SUPPORTED claims, `corrected_fact` is extracted only if the evidence states the correct value.
  - **FEVER mapping** (used in evaluation): SUPPORTED→SUPPORTS; CONTRADICTED→REFUTES; INSUFFICIENT / CONFLICTING / PARTIAL→NOT ENOUGH INFO. The PARTIAL→NEI choice is debatable; report the confusion matrix so examiners can see its effect.
- **Test:** FEVER dev in claim mode (100 claims): label accuracy and macro-F1 of the verifier *alone*, compared with the M4 RAG FEVER number; a unit test of the citation post-check.
- **Done when:** FEVER dev macro-F1 is at least RAG baseline + 3 points (if not, iterate on the prompt and evidence selection here, on dev only).

### M8 — Confidence v0 + Response Synthesis Agent (Week 11, first half)
- **Maps to:** 9.9 (v0), 9.10, O7 (partial), O8, D10.
- **Depends on:** M7.
- **Input → Output:** verdicts + fused evidence → `dict[claim_id, ConfidenceScore]` + response confidence → `FinalResponse`.
- **Key decisions:**
  - **Confidence v0:** a hand-weighted score over the feature dict `{max_entail, max_contradict, support_mass, contradict_mass, n_sources_supporting, agreement, top_relevance, verbal_confidence, label_onehot…, logic_penalty(=0 for now)}`. Save **all** features to the trace; M13 trains on them.
  - Response confidence = the minimum over the claims that bear on `short_answer` (claims mentioning the short answer string or its entities), and otherwise the mean.
  - Bands come from config cutoffs 0.90 / 0.75 / 0.50 (illustrative until M13).
  - **Synthesis rules (applied in order):**
    1. If the response band is "abstain", output an abstention template that states what is uncertain.
    2. Keep SUPPORTED claims.
    3. Replace CONTRADICTED claims with `corrected_fact` if present, otherwise drop them.
    4. Hedge PARTIAL / CONFLICTING claims ("sources disagree on…").
    5. Drop INSUFFICIENT claims.
    6. The agent LLM rewrites the kept claims into fluent text **using only the provided claims** and adds citation markers.
    7. `short_answer` is recomputed: if the original short answer's supporting claim was corrected, use the corrected value.
  - In FEVER mode: `fever_label` is taken from the claim verdict; abstention = NEI.
- **Test:** unit tests of each rule using synthetic verdicts (no LLM, mocked rewrite); a faithfulness spot-check that the final text introduces no new facts (judge on 20 outputs).
- **Done when:** all rules are covered by tests and the 20-output faithfulness check has zero unsupported additions.

### M9 — Orchestration (LangGraph) + Tracing → **MVP MILESTONE** (Week 11, second half)
- **Maps to:** D9, the MVP.
- **Depends on:** M2, M5–M8.
- **Input → Output:** `UserQuery` → `PipelineTrace`.
- **Key decisions:**
  - `orchestration/graph.py` builds a `StateGraph(PipelineState)` with nodes generate → extract → retrieve_and_build_graph → fuse → verify → logic (no-op stub until M12) → confidence → synthesize. A conditional edge after `logic` loops back to retrieval when `trigger_reretrieval` is set and `iteration < max_iterations` (default 1).
  - Hierarchy framing for the report: a *Coordinator* (the graph and its router) sits above three tiers: Evidence tier (extraction, retrieval, graph), Judgment tier (fusion, verification, logic), and Output tier (confidence, synthesis).
  - Per-claim work (query generation, retrieval, NLI) is parallelized with `asyncio` inside nodes, with bounded concurrency to respect rate limits.
  - Ablation switches in config: `graph.enabled`, `fusion.mode ∈ {stub, full}`, `logic.enabled`, `confidence.mode ∈ {none, heuristic, calibrated}` (none = always answer), `verification.mode ∈ {hierarchical, single_agent}` (single_agent = one LLM call that sees the answer and the raw retrieved passages and outputs a corrected answer; this is your hierarchy ablation), `retrieval.iterative ∈ {true, false}`.
  - A CLI `hmafact verify "question"` prints a readable trace summary.
  - Any stage exception is recorded in `state.errors`; the pipeline then degrades to returning the RAG-style answer flagged `abstained=false, response_confidence=0` rather than crashing an evaluation run.
- **Test:** an end-to-end run on 10 dev items per dataset; trace validation; a run with every ablation switch completes.
- **Done when:** the **MVP definition in §6** is met.

### M10 — Dashboard v1: Live Demo View (Week 12, first half)
- See §5. It depends on M9 traces, and the results view can already show the M4 baseline table.
- **Done when:** a saved trace can be replayed and a live query can be run, with every stage panel populated.

### M11 — Evidence Fusion Agent (full) + Wikidata Source (Week 12, second half)
- **Maps to:** 9.6, O4.
- **Depends on:** M7 (it replaces the stub; the schema is unchanged).
- **Key decisions:**
  - Add S3 **Wikidata**, triggered only for `temporal` and `numeric` claims and for entity claims with a clear subject: look up the entity's QID via the search API, fetch relevant property values (dates, quantities, locations), and render them as short synthetic passages ("Wikidata: Python (Q28865) – inception: 1991-02-20"). These become normal PASSAGE nodes from source `wikidata`.
  - **Source credibility weights** (config): wikidata 1.0 for structured date and number facts, local_wiki 0.9, wiki_api 0.85. Justify these in the report and test sensitivity in the ablations.
  - `support_mass = Σ credibility × relevance × entail`, and likewise for contradiction; *source-level* agreement means the number of distinct sources with a SUPPORTS edge compared with a CONTRADICTS edge.
  - `conflict_detected` is true when credible sources on both sides exceed a threshold. Add `CONFLICTS_WITH` edges between the passages involved.
  - `prelim_label` rules: support only → SUPPORTED; contradiction only → CONTRADICTED; both strong → CONFLICTING_EVIDENCE; neither → INSUFFICIENT_EVIDENCE. The verifier LLM makes the final call but receives the fused summary.
- **Test:** hand-built conflict fixtures (for example, a synthetic passage with a wrong date alongside a correct Wikidata fact); FEVER dev macro-F1 compared with the stub.
- **Done when:** the conflict fixtures are labelled correctly, and dev macro-F1 is no worse than the stub's (ideally better, especially on REFUTES).

### M12 — Logic Validation Agent + Iterative Re-Retrieval (Week 12→13)
- **Maps to:** 9.8, O6.
- **Depends on:** M9, M11.
- **Key decisions (two layers):**
  1. **Deterministic checks:** parse dates and numbers per claim (dateparser, regex); flag the *same entity + attribute* appearing with different values across claims (numeric/temporal); flag impossible orderings (death before birth, event before founding) using simple rules for birth/death/founded/released verbs; flag an entity mismatch when the short answer's entity doesn't appear in any SUPPORTED claim (`answer_mismatch`).
  2. **LLM cross-claim check:** one call over all claims (only when there are ≤8), asking for pairs that cannot both be true, returning `LogicIssue`s.
  - High-severity issues on claims whose label is not SUPPORTED set `trigger_reretrieval=true`. The loop regenerates queries for those claims *using the issue description* and adds the results to the graph. The logic penalty feeds into confidence.
- **Test:** 25 hand-written inconsistent claim sets (temporal, numeric, entity) and 25 consistent ones: measure detection precision and recall; confirm that re-retrieval changes the graph (the iteration count rises and new passage nodes appear).
- **Done when:** detection precision ≥0.8 on the fixtures, and the loop terminates within `max_iterations` in all dev runs.

### M13 — Confidence Calibration (Week 13, first half)
- **Maps to:** 9.9, O7.
- **Depends on:** M12 (the features are final).
- **Key decisions:**
  - Run the full pipeline on **all dev splits**. Correctness labels per claim: FEVER claims use the gold label; for HotpotQA and TruthfulQA, use the answer-level judge outcome joined to answer-bearing claims.
  - Train **logistic regression** on the feature dict (5-fold CV on dev), then apply **isotonic** calibration. Save the model to `indexes/calibrator.joblib`. `confidence.mode=calibrated` loads it.
  - Tune the band thresholds on dev to maximize selective accuracy subject to coverage ≥ 70% (the constraint prevents abstention-gaming). Write the tuned values back into config.
  - Metrics: ECE (15 bins), Brier score, AUROC of confidence against correctness, and a risk-coverage curve.
- **Done when:** dev ECE for calibrated confidence is lower than for the heuristic v0, and the reliability diagram and risk-coverage plot are saved.

### M14 — Full Evaluation + Ablations (Week 13)
- See §8. **Done when:** all test-split runs are complete, the tables are generated by script, significance tests are computed, and the failure-analysis notebook contains at least 20 categorized errors.

### M15 — Dashboard v2 (Results View) + FastAPI + Hardening (Week 14)
- Results view (see §5); FastAPI endpoints; README with a one-command demo; demo traces pre-recorded for offline presentation; code documentation for D13.
- **Done when:** a fresh clone with `uv sync` plus the downloaded index runs the demo using only a `.env` file, and the results view reproduces every table in the report.

**Weekly map:** W1–2 survey; W2–3 M0, M1; W4 M2; W5 M3; W6 M4; W7 M5; W8–9 M6; W10 M7; W11 M8, M9 (MVP); W12 M10, M11, start M12; W13 M12, M13, M14; W14 M15, report, and demo. This is tight. The cutline for slippage is in §7.

---

## 5. Dashboard

### 5.1 Architecture
Streamlit multipage app in `dashboard/`. It imports `hmafact` directly for live runs (simpler and more reliable for the demo than calling the API) and reads files from `runs/`. It never recomputes metrics itself; it only displays data the evaluation code wrote.

### 5.2 View A — Live Demo (`pages/1_Live_Demo.py`)
**Two modes:** *Live* (type a query or pick a benchmark sample, which runs the pipeline) and *Replay* (pick any saved trace from the SQLite index). **Replay is your insurance against API outages during the viva.**

**Layout, top to bottom:**
1. Query box, system selector (Vanilla / RAG / HMA-Fact / any ablation config), and a run button. While running, a stage progress indicator (`st.status`) updates as each LangGraph node completes, using LangGraph's streaming of node updates.
2. **Initial answer vs final answer**, side by side, with a text diff highlighting kept, corrected, and dropped content; a response confidence gauge; and an abstention banner when applicable.
3. **Claims table:** one row per claim, with a colored label chip, a confidence bar, and the claim type. Expanding a row shows its cited evidence passages (source badge, title, text with the matched span highlighted, NLI entail/contradict bars, retrieval score, URL), the verifier rationale, and the corrected fact.
4. **Evidence graph:** a pyvis interactive render of `GraphSnapshot`, with nodes colored by type, evidence edges colored green/red/grey for supports/contradicts/related, and a slider to step through iterations if re-retrieval happened.
5. **Logic issues panel.**
6. **Cost and latency panel:** per-stage timing (a waterfall bar), number of LLM calls, tokens, and cache hits.

**Data needed:** exactly one `PipelineTrace`, which already holds every field above. The demo needs no new logging beyond M0 and M9.

### 5.3 View B — Results (`pages/2_Results.py`)
**Controls:** select the run group (for example, `test_final`), the dataset, and the systems.
**Panels:**
1. The main comparison table (the brief's format): Accuracy, Hallucination Rate, Precision, Recall, F1, Avg Latency, plus Abstention Rate and cost per query, with 95% bootstrap CIs.
2. Grouped bar charts per metric across Vanilla / RAG / HMA-Fact for each dataset.
3. Latency box plots and an accuracy-vs-latency scatter (the RQ5 trade-off).
4. FEVER confusion matrices per system.
5. Reliability diagram and risk-coverage curve (RQ4).
6. Ablation table: the delta of each variant from full HMA-Fact.
7. **Failure explorer:** a filterable table of items where HMA-Fact was wrong or abstained. Clicking a row opens that trace in View A's replay mode.

**Data needed and where it comes from:**
- `runs/{run_id}/manifest.json`: system, config hash, git commit, models, dataset, split, timestamp.
- `runs/{run_id}/predictions.jsonl`: one row per sample with `sample_id, prediction, gold, correct, judged_truthful, abstained, confidence, latency_ms, tokens, n_llm_calls, trace_path`.
- `runs/{run_id}/metrics.json`: computed by `eval/`.
- `runs/summary.parquet`: aggregated by `scripts/report.py`, including CIs and significance tests. This is what View B reads.

### 5.4 When to build it
- **Trace schema:** fixed in M0 (this is what makes the dashboard cheap).
- **Minimal results page:** can be pointed at the M4 baseline metrics right away, but don't polish it then.
- **Dashboard v1 (demo view):** M10, immediately after the MVP. Before that, a CLI trace printout is enough; building UI before the traces stabilize means rework.
- **Dashboard v2 (results, ablations, failure explorer):** M15, after the final evaluation runs.

---

## 6. MVP-First Sequencing

**The MVP slice** is M0 → M1 → M2 → M3 → M4 → M5 → M6 (graph enabled, sources S1+S2) → M7 (fusion stub) → M8 (heuristic confidence) → M9. Logic validation is a no-op, and there is no Wikidata or calibration yet. It covers all 8 points of the brief's MVP definition: query, initial answer, atomic claims, evidence retrieval, comparison, label, confidence, and corrected/qualified answer.

**MVP is done when all of the following are true:**
1. `hmafact verify "<any factual question>"` returns a `FinalResponse` and writes a valid trace, for questions from all three datasets and for free-form questions.
2. It completes without error on 100% of a 30-item dev mini-batch (10 per dataset). Stage errors are allowed only if they degrade gracefully.
3. Median end-to-end latency is ≤ 30 s per query (cached calls excluded).
4. On the dev splits, HMA-Fact's hallucination rate is **lower than Vanilla** on at least 2 of 3 datasets. Beating RAG is *not* required at MVP stage.
5. At least one demo query visibly corrects a hallucinated claim with cited evidence. Save its trace as `demo/golden_1.json`.

If week 11 arrives and the MVP is not done, freeze all new scope and apply the cutline in §7.

---

## 7. Risks and Fallbacks

| # | Risk | Likelihood | Fallback |
|---|---|---|---|
| 1 | **NVIDIA Build credits or rate limits run out** mid-evaluation | High | The cache means nothing is lost. Switch `provider` to OpenRouter in config (same models). Last resort: serve Llama-3.1-8B on Colab with vLLM as an OpenAI-compatible server and use it for every role (a quality drop, but it runs). |
| 2 | **Colab disconnects** during index building or NLI batches | High | All GPU jobs are resumable scripts that write shards to Drive every N items. Never do GPU work interactively in notebook cells. |
| 3 | **Corpus too large or slow** to embed | Medium | The scoped corpus is already the fallback. If it is still too slow, shrink the FEVER background distractors from 20k to 5k and state this in the report. |
| 4 | **Malformed JSON** from agents | Medium | Pydantic validation plus one repair retry in the client. If it still fails, fall back to the 70B model for that call and log it. |
| 5 | **Claim extraction over- or under-splits**, causing latency and noise | Medium | The 8-claim cap, near-duplicate merging, and the fixture test in M5. |
| 6 | **Latency explodes** (claims × queries × sources × NLI) | Medium | Async per-claim fan-out, `k=5`, 2 queries per claim, NLI batched once per query, and Wikidata only for temporal and numeric claims. Report latency honestly; the brief expects overhead. |
| 7 | **HMA-Fact doesn't beat RAG** on some dataset | Medium | This is a legitimate result. Report where it helps (usually REFUTES detection and abstention-adjusted accuracy) and where it doesn't. Use the risk-coverage curve to show value at matched coverage. A clear negative-plus-analysis result earns the analysis marks. |
| 8 | **Judge unreliability** (TruthfulQA, HotpotQA) | Medium | A cross-family judge, 50 hand-checked items in M2, and 100 double-annotated test items in M14 with Cohen's κ reported. |
| 9 | **Abstention gaming** (low hallucination rate because the system refuses to answer) | Medium | Always report abstention rate and coverage; enforce the ≥70% coverage constraint in threshold tuning; report selective accuracy at matched coverage. |
| 10 | **Provider catalog changes or deprecates a model** | Medium | Model IDs live in config; the exact ID and date are recorded in each run manifest; final runs are done within one week. |
| 11 | **Schedule slip** (the biggest risk for a sequential 2-person team) | High | **Cutline, in order of what to drop:** (a) the entity-hop expansion in M6, (b) the Wikidata source (M11 becomes credibility weighting over S1+S2 only), (c) the LLM layer of logic validation (keep deterministic rules only), (d) iterative re-retrieval, (e) HotpotQA test size reduced to 150. **Never drop:** baselines, graph, verification, confidence with abstention, evaluation, or the demo, because these carry 75+ marks. |
| 12 | **Bus factor:** one teammate unavailable for a week | Medium | Since you work together on everything, both people know every module. Keep `docs/decisions.md` updated per module so either person can continue alone. |
| 13 | **Data leakage / tuning on test** | Low but fatal | Test split files are loaded only when `--split test --final` is passed, and every such run is logged. |

---

## 8. Evaluation Plan

### 8.1 Systems compared (identical settings)
Same generator (Llama-3.1-8B), temperature 0, same prompts wherever applicable, same corpus, same test samples. The only thing that differs is the pipeline: **Vanilla**, **Standard RAG**, and **HMA-Fact (full)**, plus ablations.

### 8.2 How each benchmark is scored

| Dataset | Mode | Primary metrics | Secondary metrics |
|---|---|---|---|
| **TruthfulQA** (generation) | qa | % Truthful (judge), % Truthful∧Informative, **hallucination rate** = % untruthful among answered | abstention rate; claim-level stats (HMA only) |
| **FEVER** | claim (generator and extractor bypassed) | Label accuracy, **macro-F1**, per-class P/R/F1 | evidence Recall@5 (gold page retrieved); confusion matrix |
| **HotpotQA** (distractor) | qa | EM, token F1, judge accuracy, **hallucination rate** | supporting-fact Recall@k; bridge vs comparison breakdown |
| All | — | Avg and median latency, LLM calls, tokens, estimated cost per query | ECE, Brier, AUROC, risk-coverage (HMA-Fact and RAG) |

The brief's table columns (Accuracy, Hallucination Rate, Precision, Recall, F1, Latency) are filled like this: *Accuracy* = FEVER label accuracy / HotpotQA judge accuracy / TruthfulQA % truthful; *P/R/F1* = FEVER macro P/R/F1 and HotpotQA token-level P/R/F1. Report one table per dataset rather than forcing a single mixed table. This avoids comparing unlike metrics and is easier to defend.

**Confidence for the baselines:** Vanilla and RAG get a verbalized-confidence prompt ("rate 0–1"), so calibration can be compared across all three systems.

### 8.3 Sample sizes

| Level | Per dataset (test) | Total LLM-pipeline runs* | What it supports |
|---|---|---|---|
| **Undergraduate submission (plan of record)** | 300 (FEVER balanced 100/100/100) + 100 dev | 3 systems × 900 test + ablations on a 150-item subset per dataset (6 variants) | 95% CI of about ±5–6 points on proportions; differences of about 8+ points detectable. Enough for a clear, honest result. |
| **Minimum if the schedule slips** | 150 | — | Directional result only; say so explicitly. |
| **Publication attempt** | TruthfulQA all 817; FEVER ≥1,000 (or full paper_dev) with the evidence-aware FEVER score; HotpotQA ≥1,000 | plus 2 more generator models (e.g. one different family, one larger) | Plus: ablations on full sets, 3 seeds or prompt variants, human evaluation of 200 claims, cost reporting, comparison with at least one published method (CoVe or CRAG reimplemented), and a non-scoped corpus or an explicit corpus-sensitivity study. |

*Ablations reuse cached stages (for example, "no logic" reuses every call before the logic stage), so each ablation costs far less than a full run.

### 8.4 Statistical testing
- Paired bootstrap (10,000 resamples) for differences in accuracy, F1, and hallucination rate between systems on the same items; report 95% CIs.
- McNemar's test for paired correctness (HMA-Fact vs RAG, HMA-Fact vs Vanilla).
- Judge validity: 100 test items double-annotated by both teammates (blind to system), with Cohen's κ between each person and the judge.

### 8.5 Compute and cost budget (rough; verify current prices)
- HMA-Fact: about 8–12 agent-LLM calls and about 10–15k tokens per query. 900 test queries → roughly 10–14M tokens; dev and calibration runs add about 1.5×; ablations are mostly cache hits.
- Total: roughly 25–40M tokens over the project. On NVIDIA Build this may fit within trial credits depending on your allocation. On OpenRouter at current open-weight 70B prices this is typically in the tens of US dollars, not hundreds — **check the live price pages before committing.**
- GPU: about 6–10 Colab hours total (index building and batched NLI). Vanilla and RAG runs are cheap.
- **Time:** at a conservative ~20 requests per minute, the full test sweep of all three systems is about one to two days of wall-clock time. Start M14 runs no later than early Week 13, run them overnight, and resume from cache after any interruption.

### 8.6 Ablations (map to research questions)
Run on the 150-item test subset per dataset:
| Variant | Config switch | Answers |
|---|---|---|
| − Dynamic graph (flat lists) | `graph.enabled=false` | RQ2 |
| − Fusion (stub) | `fusion.mode=stub` | RQ3 |
| − Logic validation | `logic.enabled=false` | contribution of O6 |
| − Confidence / abstention | `confidence.mode=none` | RQ4 |
| Single-agent verification | `verification.mode=single_agent` | RQ1 (hierarchy) |
| Static retrieval | `retrieval.iterative=false` | dynamic vs static |

### 8.7 Failure analysis (feeds the "Analysis and Innovation" marks)
Categorize at least 20 HMA-Fact errors into: extraction error, retrieval miss, NLI or verifier error, fusion or conflict error, synthesis drift, judge error, or benchmark ambiguity. A stage-attribution table built from the traces ("X% of residual errors originate in retrieval") is a strong analytical contribution and is cheap because everything is traced.

---

## Appendix — Handoff Template for Per-Module Specs

When you come back for a module spec, ask for it with this structure so it stays consistent with this plan:

> "Using HMA-Fact Master Plan v1.0 as the source of truth, write the Claude Code build spec for **M<n> – <name>**. Include: files to create; the exact functions and signatures; the schemas used (unchanged from §3.3 unless you say otherwise); prompts (full text); config keys added; tests (with fixtures); the acceptance criteria from §4; and anything out of scope."

Paste the plan, or at minimum §0, §2, §3.3, and the relevant §4 entry, into that request and into Claude Code's context (e.g. as `docs/PLAN.md` plus a `CLAUDE.md` pointing to it).
