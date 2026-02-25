# Methodology (Open Version)

This repository implements an evaluation loop for RAG systems where each evaluated row corresponds to a single turn.

For each turn:
1. Read structured inputs: query, answer, contexts
2. Construct a deterministic judge prompt
3. Invoke a judge (heuristic or LLM)
4. Validate strict JSON output (schema-checked)
5. Aggregate metric scores and export CSV

## Metrics (0-5)
- Faithfulness
- Context utilization
- Retrieval correctness
- Answer helpfulness
