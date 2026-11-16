"""Config loading and small JSON / JSONL helpers shared by every assist module."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "assist/configs/industreal.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    """The dataset config with every ``paths`` entry made absolute (relative to the repository root)."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["paths"] = {k: (ROOT / v).resolve() if "{" not in v else str(ROOT / v) for k, v in cfg["paths"].items()}
    cfg["graph_path"] = ROOT / cfg["graph"]
    return cfg


def load_yaml(path: Path | str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    tmp.replace(path)          # only complete files ever exist, so reruns can skip them safely


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
