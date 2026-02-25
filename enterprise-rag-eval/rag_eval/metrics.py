from __future__ import annotations
from .schemas import JudgeScores

def overall_score(scores: JudgeScores) -> float:
    # Simple average of 4 metrics on 0-5 scale
    return round(
        (scores.faithfulness + scores.context_utilization + scores.retrieval_correctness + scores.answer_helpfulness) / 4.0,
        3,
    )
