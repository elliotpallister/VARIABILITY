from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fano_project.config import Config


def analysis_key(cfg: Config) -> str:

    payload = {
        "analysis": {
            "areas": getattr(cfg.analysis, "areas", None),
            "n_runs": getattr(cfg.analysis, "n_runs", None),
            "time_sample": getattr(cfg.analysis, "time_sample", None),
        },
        "preprocessing": {
            "bin_size": cfg.preprocessing.bin_size,
            "windows": cfg.preprocessing.windows,
        },
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def analysis_cache_dir(cfg: Config) -> Path:
    key = analysis_key(cfg)
    root = Path(cfg.paths.cache) / "analysis" / key
    root.mkdir(parents=True, exist_ok=True)
    return root


def analysis_session_dir(cfg: Config, sid: int) -> Path:
    root = analysis_cache_dir(cfg)
    sdir = root / f"{sid}"
    sdir.mkdir(parents=True, exist_ok=True)
    return sdir


def analysis_cache_paths(cfg: Config, session_id: int) -> dict[str, Path]:
    sdir = analysis_session_dir(cfg, session_id)
    return {
        "mv": sdir / "mv.pkl",
        "fano": sdir / "fano.pkl",
        "metrics": sdir / "metrics.parquet",
        "summary_pkl": sdir / "summary.pkl",
        "tpc_ts": sdir / "tpc_ts.npy",
    }


def save_data(path: Path, obj: Any) -> Any:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(f"Expected DataFrame for {path.name}, got {type(obj)}")
        obj.to_parquet(path)
        return obj

    if suffix == ".npy":
        np.save(path, obj, allow_pickle=False)
        return obj

    if suffix in {".pkl", ".pickle"}:
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        return obj

    raise ValueError(f"Unsupported suffix: {suffix} for {path}")


def load_data(path: Path) -> Any:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".npy":
        return np.load(path, allow_pickle=False)

    if suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            return pickle.load(f)

    raise ValueError(f"Unsupported suffix: {suffix} for {path}")
