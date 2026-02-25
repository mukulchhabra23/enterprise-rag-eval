from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any

def read_dataset(path: str, fmt: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p.resolve()}")

    fmt = fmt.lower().strip()
    if fmt == "json":
        return json.loads(p.read_text(encoding="utf-8"))
    if fmt == "jsonl":
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows
    raise ValueError(f"Unsupported dataset format: {fmt}. Use 'json' or 'jsonl'.")

def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))
