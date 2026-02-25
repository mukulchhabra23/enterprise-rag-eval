from __future__ import annotations
from typing import List

JUDGE_SYSTEM = (
    "You are a strict evaluator for enterprise Retrieval-Augmented Generation (RAG). "
    "You must return ONLY valid JSON matching the required schema. "
    "Scores are integers 0-5. Keep rationale short."
)

def build_judge_user_prompt(
    user_query: str,
    model_answer: str,
    retrieved_contexts: List[str],
) -> str:
    ctx = "\n\n---\n\n".join(retrieved_contexts) if retrieved_contexts else "(no retrieved context provided)"
    return f"""Evaluate the RAG response.

User query:
{user_query}

Model answer:
{model_answer}

Retrieved contexts:
{ctx}

Return JSON with keys:
faithfulness, context_utilization, retrieval_correctness, answer_helpfulness, rationale

Definitions:
- faithfulness: answer is grounded in provided contexts (0=hallucinated, 5=fully grounded)
- context_utilization: uses the provided contexts effectively (0=ignores, 5=uses well)
- retrieval_correctness: retrieved contexts are relevant to query (0=irrelevant, 5=highly relevant)
- answer_helpfulness: actionable and correct given query (0=unhelpful, 5=excellent)

Output ONLY JSON.
"""
