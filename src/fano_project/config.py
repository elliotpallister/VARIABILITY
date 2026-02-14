from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

# HELPER FOR MISSING CONFIG KEYS

def _req(d: dict, key: str, ctx: str) -> Any:
  if key not in d:
    raise KeyError(
      f"Missing {key} in {ctx}. Found keys: {d.keys()}"
    )
  return d[key]

# 1: PATH CONFIGURATION

@dataclass(frozen=True)
class PathConfig:
  raw: dict
  root: Path

  @property
  def dataset(self) -> Path:
    p = Path(_req(self.raw, "data", "paths"))
    return self.root / p
  
  @property
  def cache(self) -> Path:
    p = Path(_req(self.raw, "cache", "paths"))
    return self.root / p
  
  @property
  def results(self) -> Path:
    p = Path(_req(self.raw, "results", "paths"))
    return self.root / p
  
# 2: PREPROCESSING CONFIGURATION
  
@dataclass(frozen=True)
class PreprocessingConfig:
  raw: dict

  @property
  def quality_metrics(self) -> dict:
    filters = _req(self.raw, "filters", "preprocessing")
    quality = _req(filters, "quality", "preprocessing.filters")
    return dict(quality)
  
  @property
  def effect_threshold(self) -> float:
    filters = _req(self.raw, "filters", "preprocessing")
    visual_sel = _req(filters, "visual_selectivity", "preprocessing.filters")
    effect_threshold = _req(visual_sel, "effect_threshold", "preprocessing.filters.visual_selectivity")
    return float(effect_threshold)
  
  @property
  def alpha(self) -> float:
    filters = _req(self.raw, "filters", "preprocessing")
    visual_sel = _req(filters, "visual_selectivity", "preprocessing.filters")
    alpha = _req(visual_sel, "alpha", "preprocessing.filters.visual_selectivity")
    return float(alpha)
  
  @property
  def min_n_eff(self) -> float:
    filters = _req(self.raw, "filters", "preprocessing")
    visual_sel = _req(filters, "visual_selectivity", "preprocessing.filters")
    min_n_eff = _req(visual_sel, "min_n_eff", "preprocessing.filters.visual_selectivity")
    return float(min_n_eff)
  
  @property
  def bin_size(self) -> float:
    psth = _req(self.raw, "psth", "preprocessing")
    bin_size = _req(psth, "bin_size", "preprocessing.psth")
    return float(bin_size)
  
  @property
  def windows(self) -> dict:
    psth = _req(self.raw, "psth", "preprocessing")
    windows = _req(psth, "windows", "preprocessing.psth")
    return dict(windows)
  
# 3: ANALYSIS CONFIGURATION

@dataclass(frozen=True)
class AnalysisConfig:
  raw: dict

  @property
  def window_size(self) -> dict:
    time_sampling = _req(self.raw, "time_sampling", "analysis")
    window_size = _req(time_sampling, "window_size", "analysis.time_sampling")
    return float(window_size)
  
  @property
  def areas(self) -> list:
    areas = _req(self.raw, "areas", "analysis")
    return list(areas)
  
  @property
  def n_runs(self) -> int:
    mean_matching = _req(self.raw, "mean_matching", "analysis")
    n_runs = _req(mean_matching, "n_runs", "analysis.mean_matching")
    return int(n_runs)

# MAIN CONFIGURATION

@dataclass(frozen=True)
class Config:
  raw: dict
  root: Path

  @property
  def paths(self) -> PathConfig:
    return PathConfig(_req(self.raw, "paths", "root"), self.root)
  
  @property
  def preprocessing(self) -> PreprocessingConfig:
    return PreprocessingConfig(_req(self.raw, "preprocessing", "root"))
  
  @property
  def analysis(self) -> AnalysisConfig:
    return AnalysisConfig(_req(self.raw, "analysis", "root"))


# LOADING CONFIG

def load_config(config_path: Path) -> Config:

  config_path = config_path.resolve()

  with open(config_path, "r") as f:
    raw = yaml.safe_load(f)

  root = config_path.parent.parent

  return Config(raw=raw, root=root)

