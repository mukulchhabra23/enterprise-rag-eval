# Enterprise RAG Evaluation Framework

A research-oriented framework for evaluating Retrieval-Augmented Generation (RAG) systems using **case-aware metrics** and **LLM-as-a-judge** scoring.

This repo is designed to be:
- easy to run locally (with a deterministic fallback judge), and
- extensible for real judge models (e.g., OpenAI-compatible APIs).

## What you get
- Config-driven evaluation pipeline
- Strict JSON judge outputs (validated)
- Multi-metric scoring (faithfulness, context utilization, retrieval correctness, helpfulness)
- Batch evaluation on JSONL/JSON datasets
- Example dataset + notebook
- Basic unit tests

## Quickstart

```bash
pip install -r requirements.txt
python examples/run_eval.py
```

### Using a real judge model (optional)
If you have an OpenAI-compatible API key set, the pipeline can use it:

```bash
export OPENAI_API_KEY="..."
```

Then set `judge.mode: openai` in `configs/default.yaml`.

> If no API key is present, it will automatically fall back to `judge.mode: heuristic` so you can run end-to-end.

## Repo structure
```
enterprise-rag-eval/
├── rag_eval/               # library code
├── configs/                # yaml configs
├── examples/               # runnable scripts + sample dataset
├── notebooks/              # demo notebook
├── docs/                   # methodology docs
└── tests/                  # unit tests
```

## Citation
If you use this framework in your work, please cite the associated paper/preprint (add link here).

## License
MIT
