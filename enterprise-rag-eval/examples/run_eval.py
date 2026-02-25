from rag_eval.pipeline import EvaluationPipeline

if __name__ == "__main__":
    pipeline = EvaluationPipeline(config_path="configs/default.yaml")
    df = pipeline.run("examples/sample_dataset.json")
    print("\nPreview:")
    print(df.head())
