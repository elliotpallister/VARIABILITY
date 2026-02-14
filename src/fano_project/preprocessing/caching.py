from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from fano_project.config import Config



# CACHE GENERATOR FUNCTIONS

# 1: GENERATE CACHE KEY

def preprocessing_key(cfg: Config) -> str:

  payload = {
        "psth": {
            "bin_size": cfg.preprocessing.bin_size,
            "windows": cfg.preprocessing.windows,
        },
    }

  blob = json.dumps(payload, sort_keys=True)

  return hashlib.sha1(blob.encode()).hexdigest()[:10]

# 2: MAKING PREPROCESSING CACHE DIRECTORY

def preprocessing_cache_dir(cfg: Config) -> Path:

  key = preprocessing_key(cfg)
  root = Path(cfg.paths.cache) / "preprocessing" / key
  root.mkdir(parents=True, exist_ok=True)

  return root

# 3: MAKING SESSION CACHE DIRECTORY

def session_cache_dir(cfg: Config, sid: int) -> Path:

  root = preprocessing_cache_dir(cfg)
  sdir = root / f"{sid}"
  sdir.mkdir(parents=True, exist_ok=True)
  
  return sdir

# 4: STORING CACHE PATHS

def cache_paths(cfg: Config, session_id: int) -> dict[str, Path]:
    sdir = session_cache_dir(cfg, session_id)
    return {
        "onsets": sdir / "onsets.parquet",
        "units":  sdir / "units.parquet",
        "wilcoxon": sdir / "wilcoxon.parquet",
        "psth":   sdir / "psth.npy",
        "running": sdir / "running.npy",
        "pupil": sdir / "pupil.npy",
        "licking": sdir / "licking.npy"
    }

# CACHE I/O FUNCTIONS

# 1: SAVING DATA

def save_data(
    path: Path,
    obj: Any,
 ) -> None:
   
   suffix = path.suffix.lower()

   if suffix == ".parquet":
      obj.to_parquet(path)
      return obj
   if suffix == ".npy":
      np.save(path, obj)
      return obj
   if suffix in {".pkl", ".pickle"}:
      with open(path, "wb") as f:
        pickle.dump(obj, f) 
      return obj  

# 2: LOADING DATA

def load_data(path: Path) -> Any:
   
   suffix = path.suffix.lower()

   if suffix == ".parquet":
     return pd.read_parquet(path)
   if suffix == ".npy":
     return np.load(path, allow_pickle=False)
   if suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            return pickle.load(f)
   