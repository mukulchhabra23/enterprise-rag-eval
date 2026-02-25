from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path

import yaml
import pandas as pd
from tqdm import tqdm

from .schemas import TurnInput, EvalRow
from .utils import read_dataset, ensure_dir
from .judge import make_judge
from .metrics import overall_score

@dataclass
class PipelineConfig:
    dataset_path: str
    dataset_format: str
    output_dir: str
    output_file: str
    judge_mode: str
    judge_model: str
    temperature: float
    max_tokens: int
    verbose: bool

class EvaluationPipeline:
    def __init__(self, config_path: str = "configs/default.yaml"):
        cfg = self._load_config(config_path)
        self.cfg = cfg
        self.judge = make_judge(
            mode=cfg.judge_mode,
            model=cfg.judge_model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    def _load_config(self, path: str) -> PipelineConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

        return PipelineConfig(
            dataset_path=raw["dataset"]["path"],
            dataset_format=raw["dataset"].get("format", "json"),
            output_dir=raw["output"].get("dir", "outputs"),
            output_file=raw["output"].get("file", "scores.csv"),
            judge_mode=raw["judge"].get("mode", "heuristic"),
            judge_model=raw["judge"].get("model", "gpt-4.1-mini"),
            temperature=float(raw["judge"].get("temperature", 0.0)),
            max_tokens=int(raw["judge"].get("max_tokens", 600)),
            verbose=bool(raw.get("logging", {}).get("verbose", True)),
        )

    def run(self, dataset_path: Optional[str] = None) -> pd.DataFrame:
        dpath = dataset_path or self.cfg.dataset_path
        rows = read_dataset(dpath, self.cfg.dataset_format)
        parsed: List[TurnInput] = [TurnInput(**r) for r in rows]

        out_rows: List[Dict[str, Any]] = []
        it = tqdm(parsed, desc="Scoring turns", disable=not self.cfg.verbose)

        for t in it:
            scores = self.judge.score(t.user_query, t.model_answer, t.retrieved_contexts)
            overall = overall_score(scores)
            out = EvalRow(
                case_id=t.case_id,
                turn_id=t.turn_id,
                judge_mode=self.judge.mode,
                model_used=self.judge.model_used,
                faithfulness=int(scores.faithfulness),
                context_utilization=int(scores.context_utilization),
                retrieval_correctness=int(scores.retrieval_correctness),
                answer_helpfulness=int(scores.answer_helpfulness),
                overall=float(overall),
                rationale=scores.rationale,
            )
            out_rows.append(out.model_dump())

        df = pd.DataFrame(out_rows)

        outdir = ensure_dir(self.cfg.output_dir)
        outpath = outdir / self.cfg.output_file
        df.to_csv(outpath, index=False)

        if self.cfg.verbose:
            print(f"Saved: {outpath.resolve()}")
            print(df.describe(numeric_only=True))

        return df
