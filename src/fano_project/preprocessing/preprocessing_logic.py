import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass

from fano_project.preprocessing.loading import load_session
from fano_project.preprocessing.caching import cache_paths, load_data, save_data
from fano_project.preprocessing.wilcoxon import build_vsel_table, apply_vsel_thresholds
from fano_project.config import Config

import logging

logger = logging.getLogger(__name__)

# DATACLASSES

@dataclass(frozen=True)
class PreprocessedSession:
   session_id: int
   o_table: pd.DataFrame
   u_table: pd.DataFrame
   v_table_thr: pd.DataFrame
   running: np.ndarray
   pupil: np.ndarray
   licking: np.ndarray
   psth: np.ndarray
   tpc: np.ndarray
   qc_ok: list


# HELPER FUNCTIONS

def interpolate_behaviour(
    t: list,
    b: list,
    tpc: np.ndarray
  ):

  aligned = np.stack([
    np.interp(tpc, t_i, b_i, left=np.nan, right=np.nan,)
    for t_i, b_i in zip(t, b)
  ], axis=0)


  return aligned

def clean_running(
      aligned: np.ndarray
  ):
   
  aligned = aligned[:,:-1]
  idx_with_nan = [i for i, arr in enumerate(aligned) if np.isnan(arr).any()]
  clean_aligned = np.delete(aligned, idx_with_nan, axis=0)

  return clean_aligned, idx_with_nan


# LOGIC FUNCTIONS

# 1: BUILDING ONSETS TABLE

def build_onsets_table(
    s_table: pd.DataFrame,
    t_table: pd.DataFrame,
  ) -> pd.DataFrame:

  # 1.1 Cleaning and sorting tables

  s = s_table.copy()
  t = t_table.copy()

  s["start_time"] = pd.to_numeric(s["start_time"], errors="coerce")
  t["start_time"] = pd.to_numeric(t["start_time"], errors="coerce")
  t["stop_time"]  = pd.to_numeric(t["stop_time"], errors="coerce")

  s = s.dropna(subset=["start_time"])
  t = t.dropna(subset=["start_time", "stop_time"])

  s = s.sort_values("start_time").reset_index(drop=True)
  t = t.sort_values("start_time").reset_index().rename(columns={"trials_id": "trial_id"})

  active = s[s["active"]].copy().reset_index(drop=True)
  passive = s[~s["active"]].copy().reset_index(drop=True)

  # 1.2 Building shared presentation index for active and passive stimulus presentations


  active["pres_idx"] = np.arange(len(active))
  passive["pres_idx"] = np.arange(len(passive))

  if len(active) != len(passive):
        raise ValueError(
            f"Active ({len(active)}) and passive ({len(passive)}) counts differ; "
            "cannot map by order safely."
        )

  # 1.3 Assign trial_id to active presentations

  active_assigned = pd.merge_asof(
        active,
        t[["trial_id", "start_time", "stop_time"]]
          .rename(columns={"start_time": "trial_start", "stop_time": "trial_end"}),
        left_on="start_time",
        right_on="trial_start",
        direction="backward",
        allow_exact_matches=True,
    )
  
  in_trial = (active_assigned["start_time"] >= active_assigned["trial_start"]) & (active_assigned["start_time"] <  active_assigned["trial_end"])
  active_assigned.loc[~in_trial, "trial_id"] = np.nan
  active_assigned = active_assigned.dropna(subset=["trial_id"]).copy()
  active_assigned["trial_id"] = active_assigned["trial_id"].astype(int)

  active_assigned = active_assigned.sort_values(["trial_id", "start_time", "pres_idx"])
  active_assigned["trial_frame"] = active_assigned.groupby("trial_id").cumcount()

  # 1.4 Mapping columns from active to passive

  map_cols = active_assigned[["pres_idx", "trial_id", "trial_frame"]]
  passive_assigned = passive.merge(map_cols, on="pres_idx", how="left")

  # 1.5 Merging passive and active columns

  out = pd.concat([active_assigned, passive_assigned], ignore_index=True)
  out = out.sort_values("start_time").reset_index(drop=True)

  cols_to_save = ["image_name", "active", "trial_id", "trial_frame", "is_change", "start_time"]

  out = out.dropna(subset=["trial_id", "trial_frame"]).copy()
  out["trial_id"] = out["trial_id"].astype(int)
  out["trial_frame"] = out["trial_frame"].astype(int)

  logger.info(f"Onset table built. {len(out)} onsets in table over {max(out['trial_id'])} trials")
  
  return out[cols_to_save]

# 2: BUILDING PSTHS

def build_psths(
        cfg: Config, 
        o_table: pd.DataFrame,
        u_table: pd.DataFrame,
        spikes: dict
  ) -> np.ndarray:
    
    # 2.1 Collecting timing data

    windows = cfg.preprocessing.windows
    bin_size = cfg.preprocessing.bin_size

    start = windows["pre"][0]
    end = windows["post"][1]
    duration = end - start

    bins = np.arange(0, duration+bin_size, bin_size)
    tpc = (bins[:-1] + bins[1:]) / 2
    tpc = tpc + start

    # 2.2 Defining output structure

    n_units = len(u_table)
    n_onsets = len(o_table)
    n_timepoints = len(tpc)

    psth = np.zeros((n_units, n_onsets, n_timepoints), dtype=np.float32)

    # 2.3 Collecting onsets and unit indices

    onsets = o_table["start_time"].to_numpy()
    uids = u_table.index.to_list()

    pbar = tqdm(total=n_units * n_onsets, desc="Building PSTHs", unit="psth")

    # 2.4 Making PSTH

    for u, uid in enumerate(uids):

      unit_spikes = spikes[uid]

      if pbar is not None:
        pbar.set_postfix(unit_id=uid)

      for o, onset in enumerate(onsets):

        si = np.searchsorted(unit_spikes, onset+start)
        ei = np.searchsorted(unit_spikes, onset+end)
        sts = unit_spikes[si:ei] - onset
        trace = np.histogram(sts-start, bins)[0]

        psth[u, o, :] = trace

        if pbar is not None:
          pbar.update(1)
    
    pbar.close()

    logger.info(f"Built PSTH cube of {psth.shape[0]} units over {psth.shape[1]} onsets, with spike times binned in {bin_size} bins between {start} and {end} for each onset")

    return psth

# 3: BUILDING BEHAVIOUR

def build_behaviour(
   cfg: Config,
   o_table: pd.DataFrame,
   dat: pd.DataFrame,
   *,
   type: str
 ):
  
  # COLLECTING TIMING DATA
   
  windows = cfg.preprocessing.windows
  bin_size = cfg.preprocessing.bin_size

  start = windows["pre"][0]
  end = windows["post"][1]
  duration = end - start

  bins = np.arange(0, duration+bin_size, bin_size)
  tpc = (bins[:-1] + bins[1:]) / 2
  tpc = tpc + start

  onsets = o_table["start_time"].to_numpy()

  # BUILDING DATA STRUCTURE

  if type == "running":
     query = "speed"

  elif type == "pupil":
     query = "pupil_area"

  elif type == "licking":

    behaviour = np.zeros((len(onsets), len(tpc)), dtype=np.float32)
    l_raw = dat["timestamps"].to_numpy()

    for i, o in enumerate(onsets):

      si = np.searchsorted(l_raw, o+start)
      ei = np.searchsorted(l_raw, o+end)
      l_aligned = l_raw[si:ei] - o

      l = np.histogram(l_aligned - start, bins)[0]

      behaviour[i,:] = l
    
    return behaviour
  else:
    raise ValueError(f"Unknown type={type!r}. Expected 'running', 'pupil', or 'licking'.")

  t_all = []
  b_all = []

  t_raw = dat["timestamps"].to_numpy()
  b_raw = dat[query].to_numpy()

  for o in onsets:
      t = t_raw - o
      mask = (t >= start) & (t < end)
      t_all.append(t[mask])
      b_all.append(b_raw[mask])

  behaviour = interpolate_behaviour(t_all, b_all, tpc)

  return behaviour

# 4: FILTER UNTIS

def quality_filter(
      cfg: Config, 
      u_table: pd.DataFrame,
    ) -> list:
  
  u = u_table.copy()
   
  # 3.1 Collect metrics from configuration

  quality = cfg.preprocessing.quality_metrics

  snr = quality["snr"]
  isi = quality["isi_violations"]
  fr = quality["firing_rate"]

  # 3.2 Construct mask

  qc_ok = (
    (u["snr"].to_numpy() >= snr)
    & (u["isi_violations"].to_numpy() < isi)
    & (u["firing_rate"].to_numpy() > fr)
  )

  return qc_ok


# MAIN

def preprocess_session(cfg: Config, session_id: int) -> PreprocessedSession:
    
   
    # COLLECTING CACHE PATHS

    p = cache_paths(cfg, session_id)

    exists = {k: v.exists() for k, v in p.items()}
    true_keys, false_keys = (
        [k for k, ok in exists.items() if ok],
        [k for k, ok in exists.items() if not ok],
      )
    
    logger.info(f"[Preprocessing] Existing cached files: {true_keys}")
    logger.info(f"[Preprocessing] Missing cached files: {false_keys}")

    session_dict = None


    # LOADING ALLEN DATA IF ANY CACHE FILES MISSING

    if not all(exists.values()):
      session_dict = load_session(cfg, session_id)


    # CACHE I/O --> CHECKING FOR CACHED DATA, IF NOT AVAILABLE, BUILDING AND SAVING

    o_table = load_data(p["onsets"]) if p["onsets"].exists() else save_data(p["onsets"], build_onsets_table(session_dict["s_table"], session_dict["t_table"]))
    logger.info("[Preprocessing] Onsets table loaded.")

    u_table = load_data(p["units"]) if p["units"].exists() else save_data(p["units"], session_dict["u_table"])
    logger.info("[Preprocessing] Unit table loaded.")

    psth = load_data(p["psth"]) if p["psth"].exists()  else save_data(p["psth"], build_psths(cfg, o_table, u_table, session_dict["spikes"]))
    logger.info("[Preprocessing] PSTH array loaded.")

    running = load_data(p["running"]) if p["running"].exists() else save_data(p["running"], build_behaviour(cfg, o_table, session_dict["running"], type='running'))
    logger.info("[Preprocessing] Running array loaded.")

    pupil = load_data(p["pupil"]) if p["pupil"].exists() else save_data(p["pupil"], build_behaviour(cfg, o_table, session_dict["pupil"], type='pupil'))
    logger.info("[Preprocessing] Pupil array loaded.")

    licking = load_data(p["licking"]) if p["licking"].exists() else save_data(p["licking"], build_behaviour(cfg, o_table, session_dict["licks"], type='licking'))
    logger.info("[Preprocessing] Licking array loaded.")

    v_table = load_data(p["wilcoxon"]) if p["wilcoxon"].exists() else save_data(p['wilcoxon'], build_vsel_table(cfg, u_table, o_table, psth))
    logger.info("[Preprocessing] Wilcoxon visual selectivity table loaded.")

    del session_dict


    # CONSTRUCING OUTPUT FOR ANALYSIS

    windows = cfg.preprocessing.windows
    bin_size = cfg.preprocessing.bin_size
    start = windows["pre"][0]
    end = windows["post"][1]
    duration = end - start
    bins = np.arange(0, duration+bin_size, bin_size)
    tpc = (bins[:-1] + bins[1:]) / 2
    tpc = tpc + start
    logger.info("Obtained time array")

    u_table = u_table.copy()
    u_table["pos_idx"] = np.arange(len(u_table), dtype=np.int64)
    o_table = o_table.reset_index(drop=True).copy()
    o_table["pos_idx"] = np.arange(len(o_table), dtype=np.int64)
    logger.info("Positional index applied to units and onsets")
    
    qc_ok = quality_filter(cfg, u_table)
    logger.info("Obtained quality control filter")

    v_table_thr = apply_vsel_thresholds(cfg, v_table)
    logger.info("Obtained visual selectivity filter")


    # OUTPUT
    
    return PreprocessedSession(
      session_id=session_id,
      o_table=o_table,
      u_table=u_table,
      v_table_thr=v_table_thr,
      running=running,
      pupil=pupil,
      licking=licking,
      psth=psth,
      tpc=tpc,
      qc_ok=qc_ok
    )