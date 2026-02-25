from __future__ import annotations
import os
import json
from typing import List, Optional, Dict, Any
from pydantic import ValidationError

from .schemas import JudgeScores
from .prompts import JUDGE_SYSTEM, build_judge_user_prompt
from .utils import clamp_int

class BaseJudge:
    mode: str = "base"
    model_used: str = "n/a"

    def score(self, user_query: str, model_answer: str, retrieved_contexts: List[str]) -> JudgeScores:
        raise NotImplementedError

class HeuristicJudge(BaseJudge):
    """Deterministic fallback judge so the repo runs end-to-end without API keys."""
    mode = "heuristic"
    model_used = "heuristic-v0"

    def score(self, user_query: str, model_answer: str, retrieved_contexts: List[str]) -> JudgeScores:
        uq = (user_query or "").lower()
        ans = (model_answer or "").lower()
        ctx = " ".join(retrieved_contexts).lower() if retrieved_contexts else ""

        # retrieval correctness: simple lexical overlap
        overlap = 0
        for tok in set([t for t in uq.replace("/", " ").replace("-", " ").split() if len(t) > 3]):
            if tok in ctx:
                overlap += 1
        retrieval_correctness = clamp_int(overlap, 0, 5)

        # context utilization: if answer references something from context
        util = 0
        for tok in set([t for t in ctx.split() if len(t) > 5]):
            if tok in ans:
                util += 1
                if util >= 5:
                    break
        context_utilization = clamp_int(util, 0, 5)

        # faithfulness: if answer claims things absent from context (very naive)
        # if no context, can't be grounded
        if not retrieved_contexts:
            faithfulness = 1 if len(ans) > 0 else 0
        else:
            # reward if answer uses context tokens
            faithfulness = clamp_int(context_utilization + 1, 0, 5)

        # helpfulness: basic heuristic: contains steps/bullets or "check/verify/restart"
        helpful_markers = ["step", "check", "verify", "restart", "reboot", "run", "open", "configure", "update", "install"]
        hm = sum(1 for m in helpful_markers if m in ans)
        answer_helpfulness = clamp_int(2 + hm // 2, 0, 5)

        rationale = (
            "Heuristic scoring based on token overlap and presence of troubleshooting markers. "
            "Switch to OpenAI judge for higher-fidelity scoring."
        )
        return JudgeScores(
            faithfulness=faithfulness,
            context_utilization=context_utilization,
            retrieval_correctness=retrieval_correctness,
            answer_helpfulness=answer_helpfulness,
            rationale=rationale[:500],
        )

class OpenAIJudge(BaseJudge):
    mode = "openai"

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 600):
        self.model_used = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        # Import lazily so repo can run without OpenAI installed/configured in some environments
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "OpenAI package import failed. Install dependencies with `pip install -r requirements.txt`."
            ) from e

        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Use heuristic judge or set the env var.")

    def score(self, user_query: str, model_answer: str, retrieved_contexts: List[str]) -> JudgeScores:
        user_prompt = build_judge_user_prompt(user_query, model_answer, retrieved_contexts)

        resp = self._client.chat.completions.create(
            model=self.model_used,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = resp.choices[0].message.content or ""
        # strict JSON only
        try:
            data = json.loads(content)
        except Exception as e:
            raise ValueError(f"Judge did not return valid JSON. Raw content: {content[:500]}") from e

        try:
            return JudgeScores(**data)
        except ValidationError as e:
            raise ValueError(f"Judge JSON failed schema validation: {e}") from e

def make_judge(mode: str, model: str, temperature: float, max_tokens: int) -> BaseJudge:
    mode = (mode or "").lower().strip()
    if mode == "openai":
        return OpenAIJudge(model=model, temperature=temperature, max_tokens=max_tokens)
    return HeuristicJudge()
