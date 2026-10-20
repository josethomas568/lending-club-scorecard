"""Load config.yaml and resolve paths against the repo root."""
from pathlib import Path
import json

import yaml

REPO = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = REPO / "config.yaml") -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"] = {k: str(REPO / v) for k, v in cfg["paths"].items()}
    marker = REPO / "data/raw/SYNTHETIC"
    if marker.exists():
        path, size = marker.read_text().splitlines()[:2]
        raw = Path(cfg["paths"]["raw_csv"])
        if raw.exists() and str(raw.resolve()) == path and raw.stat().st_size == int(size):
            cfg["data_source"] = "synthetic"
    for key in ("model_dir", "results_dir", "figures_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    return cfg


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def load_json(path: str | Path):
    with open(path) as fh:
        return json.load(fh)
