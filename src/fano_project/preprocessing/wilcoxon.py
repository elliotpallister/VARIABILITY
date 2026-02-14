from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata
from statsmodels.stats.multitest import fdrcorrection
from tqdm.auto import tqdm

from fano_project.config import Config

# 1: PERFORMING ONE SIDED WILCOXON TEST

def wilcoxon_one_sided(stim: np.ndarray, base: np.ndarray):
    D = stim - base
    nz = D != 0

    if not np.any(nz):
        return (1.0, "all_zero", 0, 0.0, 0.0)

    Dn = D[nz]

    try:
        stat, p = wilcoxon(
            Dn,
            zero_method="wilcox",
            alternative="greater",
            correction=False,
            mode="exact",
        )
        mode = "exact"
    except Exception:
        stat, p = wilcoxon(
            Dn,
            zero_method="wilcox",
            alternative="greater",
            correction=True,
            mode="approx",
        )
        mode = "approx"

    abs_ranks = rankdata(np.abs(Dn), method="average")
    W_plus = (abs_ranks * (Dn > 0)).sum()
    W_minus = (abs_ranks * (Dn < 0)).sum()
    denom = (W_plus + W_minus)
    r_rb = 0.0 if denom == 0 else (W_plus - W_minus) / denom

    return (float(p), mode, int(Dn.size), float(np.median(Dn)), float(r_rb))

# 2: CONSTRUCT TABLE FOR WILCOXON VISUAL SELECTIVITY RESULTS

def build_vsel_table(
    cfg: Config,
    u_table: pd.DataFrame,
    o_table: pd.DataFrame,
    psth: np.ndarray
  ) -> pd.DataFrame:

  o_table = o_table.copy()
  o_table = o_table.reset_index(drop=True)

  # 2.1 Collecting timing window data from config and converting to index

  windows = cfg.preprocessing.windows
  bin_size = cfg.preprocessing.bin_size

  baseline_window = windows["pre"]
  evoked_window = windows["evoked"]

  start = windows["pre"][0]
  end = windows["post"][1]
  duration = end - start

  bins = np.arange(0, duration+bin_size, bin_size)
  tpc = (bins[:-1] + bins[1:]) / 2
  tpc = tpc + start
  
  baseline_window = windows["pre"]
  evoked_window = windows["evoked"]

  tpc = np.array(tpc, dtype=float)
  bi = np.searchsorted(tpc, baseline_window)
  ei = np.searchsorted(tpc, evoked_window)

  # 2.2 Obtaining spike counts in baseline and evoked windows from each psth

  baseline = psth[:,:,bi[0]:bi[1]].sum(axis=2)
  evoked = psth[:,:,ei[0]:ei[1]].sum(axis=2)

  # 2.3 Looping over images in onset table and performing wilcoxon test for each unit

  rows = []

  n_groups = o_table.groupby(["image_name", "active"]).ngroups
  n_units = len(u_table.index)
  pbar = tqdm(total=n_groups * n_units, desc="Wilcoxon vsel", unit="test")

  for (image_name, active), dat in o_table.groupby(["image_name", "active"]):

    oi = dat.index.to_numpy()

    pbar.set_postfix(image=image_name, active=active)

    for u, uid in enumerate(u_table.index):

      b = baseline[u, oi]
      e = evoked[u, oi]

      p, mode, n_eff, median_diff, r_rb = wilcoxon_one_sided(e, b)

      rows.append({
        "unit_id": uid,
        "image_name": image_name,
        "active": active,
        "p": p,
        "mode": mode,
        "n_eff": n_eff,
        "median_diff": median_diff,
        "r_rb": r_rb
      })

      pbar.update(1)
    
  out = pd.DataFrame(rows)

  pbar.close()

  return out

# 3: APPLY THRESHOLDS TO VSEL FROM THE CONFIG

def apply_vsel_thresholds(
    cfg: Config,
    v_table: pd.DataFrame
  ) -> pd.DataFrames:
  
  # 3.1 Loading config data

  alpha = cfg.preprocessing.alpha
  effect_threshold = cfg.preprocessing.effect_threshold
  min_n_eff = cfg.preprocessing.min_n_eff

  # 3.2 Initialising new columns in the table

  out = v_table.copy()
  out["p_used"] = out["p"].fillna(1.0)
  out["p_adj"] = 1.0
  out["rej_fdr"] = False

  # 3.3 FDR correction

  for (_, _), dat in out.groupby(["image_name", "active"], sort=False):
      idx = dat.index
      pvals = dat["p_used"].to_numpy()

      rej, p_fdr = fdrcorrection(pvals, alpha=alpha, method="indep")
      out.loc[idx, "p_fdr"] = p_fdr
      out.loc[idx, "rej_fdr"] = rej

  # 3.4 Thresholding via effect size

  out["pass_effect"] = out["median_diff"].abs() > effect_threshold
  out["pass_neff"] = out["n_eff"] >= min_n_eff

  out["selective"] = out["rej_fdr"] & out["pass_effect"] & out["pass_neff"]

  return out


