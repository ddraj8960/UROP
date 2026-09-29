# data/samples — Benchmark Split Provenance

**Random seed:** 42  
**Dev size:** 100 per dataset (300 total)  
**Test size:** 300 per dataset (900 total)  
**Ablation subset:** 150 per dataset, flagged with `ablation_subset=true`  

## Datasets

### truthfulqa
- **Source:** truthful_qa/generation/validation
- **Resolved revision:** `0.0.0`
- **Raw rows:** 817
- **After filtering:** 817

### fever
- **Source:** https://fever.ai/download/fever/shared_task_dev.jsonl
- **Resolved revision:** `e89865bfe1b4dd054e03dd57d7241a6fde24862905f31117cf0cd719f7c78df7`
- **Raw rows:** 19998
- **After filtering:** 19998

### hotpotqa
- **Source:** https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet
- **Resolved revision:** `c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`
- **Raw rows:** 7405
- **After filtering:** 7405

## Caveats
- TruthfulQA has ~38 categories; very small categories may have 0 dev items.
- HotpotQA comparison questions are ~20% of the sample (~60 test items); comparison-specific breakdowns will be noisy.
- FEVER NLI model (DeBERTa) was trained on FEVER *train*; we evaluate on *dev*. Disclosed per master plan §2.
- HotpotQA yes/no answers (comparison type) are included in `gold_answer`.
