"""
preprocessing.py

Functions for preprocessing stimulus and unit data

Author: Elliot Pallister
"""

import numpy as np
import pandas as pd
from pathlib import Path

"""

STIMULUS & TRIAL PREPROCESSING

""" 

# CONSTRUCTING AND SAVING DATAFRAME OF ONSET METADATA

def build_onsets_table(
  session_id: int,
  s_table: pd.DataFrame,
  t_table: pd.DataFrame
  ) -> None:

  s_table = s_table.copy()

  # Aligning trial ID to stimulus onsets

  for tid, trial in t_table.iterrows():
    t_start = trial['start_time']
    t_end = trial['stop_time']

    mask = (s_table["start_time"] >= t_start) & (s_table["start_time"] < t_end)
    s_table.loc[mask, 'trial_id'] = tid

  # Assigning trial frame to stimulus onsets
  
  s_table['trial_frame'] = (
    s_table.groupby("trial_id")
           .cumcount()
  )

  # Copying trial framing to passive onsets and cleaning entries

  s_table.loc[s_table["active"] == False, ["trial_id", "trial_frame"]] = s_table.loc[s_table["active"] == True, ["trial_id", "trial_frame"]].values
  s_table["trial_id"] = s_table["trial_id"].replace(["", " ", "nan", "None"], np.nan)
  s_table = s_table.dropna(subset=["trial_id"])
  s_table[["trial_id", "trial_frame"]] = s_table[["trial_id", "trial_frame"]].astype(int)

  # Defining onset table structure

  cols_to_save = [
    "image_name",
    "active",
    "trial_id",
    "trial_frame",
    "is_change",
    "start_time",
  ]

  out = s_table[cols_to_save]

  return out

  

"""

RESPONSE PREPROCESSING

"""

def build_psths(
  units: pd.DataFrame,
  spikes: pd.DataFrame,
  o_table: pd.DataFrame,
  windows: dict,
  bin_size: int
  ) -> np.ndarray:

  # Defining common variables

  start = windows["pre"][0]
  end = windows["post"][1]
  duration = end - start
  bins = np.arange(0, duration+bin_size, bin_size)
  tps = (bins[:-1] + bins[1:]) / 2
  tps = tps + start

  # Defining output structure

  n_units = len(units)
  n_onsets = len(o_table)
  n_timepoints = len(tps)

  psth = np.zeros((n_units, n_onsets, n_timepoints), dtype=np.float32)

  # Collecting onsets and unit indices

  onsets = o_table["start_time"].to_numpy()
  uids = units.index.to_list()

  for u, uid in enumerate(uids):

    unit_spikes = spikes[uid]

    for o, onset in enumerate(onsets):

      si = np.searchsorted(unit_spikes, onset+start)
      ei = np.searchsorted(unit_spikes, onset+end)
      sts = unit_spikes[si:ei] - onset
      trace = np.histogram(sts-start, bins)[0]

      psth[u, o, :] = trace

  return psth, tps










    

  





