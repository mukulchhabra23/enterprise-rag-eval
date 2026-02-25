from rag_eval.pipeline import EvaluationPipeline

def test_pipeline_runs_heuristic(tmp_path, monkeypatch):
    # Force outputs to tmp folder
    import yaml
    from pathlib import Path

    cfg_path = tmp_path / "cfg.yaml"
    cfg = {
        "dataset": {"path": "examples/sample_dataset.json", "format": "json"},
        "output": {"dir": str(tmp_path / "out"), "file": "scores.csv"},
        "judge": {"mode": "heuristic", "model": "ignored", "temperature": 0.0, "max_tokens": 200},
        "logging": {"verbose": False},
    }
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    pipeline = EvaluationPipeline(config_path=str(cfg_path))
    df = pipeline.run("examples/sample_dataset.json")

    assert len(df) >= 1
    assert set(["faithfulness","context_utilization","retrieval_correctness","answer_helpfulness","overall"]).issubset(df.columns)
