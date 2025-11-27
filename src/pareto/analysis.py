"""
analysis.py

Main analysis functions

Author: Elliot Pallister
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Time sampling the raw PSTH trace

def time_sampled_psth(
    psths: np.ndarray,
    tps: np.ndarray,
    sample_window: float,
  ):

  window_length = int(sample_window / 10)
  n_timesamples = int(tps.shape[0] - window_length + 1)

  tpc = np.array([
    (tps[tp] + tps[tp + window_length - 1]) / 2
    for tp in range(n_timesamples)
  ])

  windowed = np.array([
    psths[:, :, i:i+window_length].sum(axis=2)
    for i in range(n_timesamples)
  ])

  return windowed, tpc, n_timesamples

# Arranging time-sampled PSTH traces into the primary data framings

def data_framing(
    ts_psths: np.ndarray,
    onsets: pd.DataFrame,
    vsel: pd.DataFrame,
    units: pd.DataFrame
  ):

  out = {}

  # Defining common variables

  areas = units["structure_acronym"].unique().tolist()
  conditions = ["active", "passive"]

  # Main framing logic

  for img, grp in onsets.groupby("image_name"):

    iw_onsets = grp.index.values
    onset_conditions = grp["active"]
    onset_frames = grp["trial_frame"]
    frame_keys = np.unique(onset_frames)

    iw_vsel = vsel[vsel["image"] == img]
    mask = (iw_vsel['reject'] & iw_vsel['pass_effect']).values
    
    iw_units = units[mask]
    iw_units_slice = iw_units.index.values

    iw_ts = ts_psths[:, iw_units_slice, :][:,:, iw_onsets]

    for c in conditions:

      if c == "passive":
        onset_conditions = ~onset_conditions

      iw_ts_cond = iw_ts[:,:,onset_conditions]

      for area in areas:

        area_mask = iw_units['structure_acronym'].values == area
        area_iw_ts_cond = iw_ts_cond[:,area_mask,:]

        if area not in out:

          out[area] = {}
        
        if c not in out[area]:

          out[area][c] = {"iw": {}, "fw": {}}

        if img not in out[area][c]["iw"]:

          out[area][c]["iw"][img] = area_iw_ts_cond

        for frame in frame_keys:

          frame_mask = onset_frames[onset_conditions] == frame
          fo = area_iw_ts_cond[:,:,frame_mask]

          if frame not in out[area][c]["fw"]:

            out[area][c]["fw"][frame] = {}
          
          out[area][c]["fw"][frame][img] = fo

  return out
        


        

      
    



# Calculating the fano factor via linear regession

def calculate_fano(
    iw_dict: dict,
    areas: list,
    n_timesamples: int
  ):

  af = {}

  for area in areas:

    means = []
    variances = []

    for image, d in iw_dict.items():

      ai = [i for i, x in enumerate(d[image]["au"]) if x == area]
      ap = d["ts_psths"][:,ai,:]

      means.append(np.mean(ap, axis=2))
      variances.append(np.var(ap, axis=2))

    means = np.hstack(means)
    variances = np.hstack(variances)

    fanos = []

    for t in range(n_timesamples):
      
      t_mean = means[t,:] 
      t_var = variances[t,:]
      slope = stats.linregress(t_mean, t_var)[0]
      fanos.append(slope)

    af[area] = np.array(fanos)

  return af

# Computing fano metrics

def compute_fano_metrics(
    fanos: np.ndarray, 
    tpc: np.ndarray,
  ):

  # Defining common variable pool

  windows = {
    "base": (-0.25, 0),
    "early": (0, 0.1),
    "late": (0.1, 0.25),
    "post": (0.25, 0.5)
  }

  # Computing values

  values = {}

  for k, w in windows.items():
    si = np.searchsorted(tpc, w[0])
    ei = np.searchsorted(tpc, w[1])

    val_fano = fanos[si:ei].mean()
    values[k] = val_fano

  # QI

  QI_early = (values["base"] - values["early"]) / (values["base"] + values["early"])
  QI_late = (values["base"] - values["late"]) / (values["base"] + values["late"])

  # RI

  RI = (values["late"] - values["early"]) / (values["late"] + values["early"])

  # AUC

  AUC_windows = windows
  del AUC_windows["base"]

  AUCs = {}

  for k, w in AUC_windows.items():

    mask = (tpc >= w[0]) & (tpc < w[1])
    y = fanos[mask] - values["base"]
    x = tpc[mask]

    AUC = np.trapz(y, x)
    AUCs[k] = AUC

  return QI_early, QI_late, RI, AUCs





  

  