# Metrics

All metrics are integers on a 0-5 scale.

- **Faithfulness**: groundedness of the answer with respect to the retrieved contexts
- **Context utilization**: whether the model used relevant parts of provided contexts
- **Retrieval correctness**: whether retrieved contexts are relevant to the query
- **Answer helpfulness**: utility/actionability/correctness given the query

The `overall` score is the mean of the four metrics.
