from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, conint

ScoreInt = conint(ge=0, le=5)

class TurnInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    case_id: str
    turn_id: str
    user_query: str
    model_answer: str
    retrieved_contexts: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

class JudgeScores(BaseModel):
    # 0-5 integer scale
    faithfulness: ScoreInt
    context_utilization: ScoreInt
    retrieval_correctness: ScoreInt
    answer_helpfulness: ScoreInt

    # Short human-readable note (kept small for privacy/traceability)
    rationale: str = Field(default="", max_length=500)

class EvalRow(BaseModel):
    case_id: str
    turn_id: str
    judge_mode: str
    model_used: str
    faithfulness: int
    context_utilization: int
    retrieval_correctness: int
    answer_helpfulness: int
    overall: float
    rationale: str
