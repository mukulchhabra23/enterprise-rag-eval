# Architecture

Core components:
- `EvaluationPipeline`: loads config, reads dataset, orchestrates scoring, writes outputs
- `Judge`: produces strict JSON scores (heuristic fallback included)
- `Schemas`: pydantic models for inputs/outputs
- `Metrics`: aggregation helpers

The design is intentionally minimal to keep the research artifact portable.
