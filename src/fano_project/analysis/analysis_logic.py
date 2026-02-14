from __future__ import annotations

import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm

from fano_project.config import Config
from fano_project.preprocessing.preprocessing_logic import PreprocessedSession
from fano_project.preprocessing.loading import get_data
from fano_project.analysis.caching import analysis_cache_paths, save_data, load_data

import logging

logger = logging.getLogger(__name__)

# DATACLASSES

@dataclass(frozen=True)
class AnalysedSession:
   session_id: int
   tpc_ts: np.ndarray
   mv: dict
   fano: dict
   metrics: pd.DataFrame
   running: np.ndarray
   pupil: np.ndarray
   summary: dict

# HELPER FUNCTIONS

def make_rngs(n: int, seed: int) -> list[np.random.Generator]:
    ss = np.random.SeedSequence(seed)
    return [np.random.default_rng(s) for s in ss.spawn(n)]


# LOGIC FUNCTIONS

# 1: TIME-SAMPLING THE PSTH

def time_sample_psth(
    cfg: Config,
    psth: np.ndarray,
    tpc: list, 
  ):

  # 1.1 Collecting data from config

  sw = cfg.analysis.window_size
  sw_len = int(sw * 100)

  # 1.2 Calculating n_timesamples

  n_timesamples = int(len(tpc) - sw_len + 1)

  # 1.3 Transforming tpc and psth

  tpc_ts = np.array([
    (tpc[tp] + tpc[tp + sw_len - 1]) / 2
    for tp in range(n_timesamples)
  ])


  psth_ts = np.array([
    psth[:, :, i:i+sw_len].sum(axis=2)
    for i in tqdm(range(n_timesamples), desc="time sampling", leave=True)
  ])

  # 1.4 Restoring U x O x T structure

  psth_ts = np.moveaxis(psth_ts, 0, 2)

  return psth_ts, tpc_ts

# 2: COMPUTING THE MEAN AND VARIANCE

def compute_mv(
    areas: list,
    psth_ts: np.ndarray,
    o_table: pd.DataFrame,
    u_table: pd.DataFrame,
    v_table_thr: pd.DataFrame,
    qc_ok: np.ndarray
  ) -> dict:

  out = {}

  groups = o_table.groupby(["image_name", "active"], sort=False)
  pbar = tqdm(total=groups.ngroups, desc="mean/var", leave=True, mininterval=0.2)

  for (image_name, active), dat in groups:
    if image_name == "omitted":
       continue

    pbar.set_postfix(image=image_name, active=active)
    for area in areas:

      # 2.1 Constructing unit mask

      area_ok = (u_table["structure_acronym"].to_numpy() == area)

      vsel_loc = (v_table_thr["image_name"] == image_name) & (v_table_thr["active"] == active)
      vsel_ok = v_table_thr.loc[vsel_loc].set_index("unit_id")["selective"]
      vsel_ok = vsel_ok.reindex(u_table.index, fill_value=False).to_numpy()

      units_ok =  qc_ok & area_ok & vsel_ok

      # 2.2 Getting unit and onset positions to slice psth

      oidx = dat.index.to_numpy().tolist()
      uidx = u_table.loc[units_ok, "pos_idx"].to_numpy().tolist()

      # 2.3 Slicing psth

      filtered_psth_ts = psth_ts[uidx,:,:][:,oidx,:]

      # 2.4 Computing mean and variance

      mean = filtered_psth_ts.mean(axis=1)
      variance = filtered_psth_ts.var(axis=1, ddof=1)

    # 2.5 Setting to output dict

      out[(area, image_name, active)] = {
        "unit_ids": u_table[units_ok].index.to_list(),
        "mean": mean,
        "variance": variance,
        "n_repeats": len(dat)
      }
    
    pbar.update(1)

  pbar.close()

  return out

# 3: COMPUTE CHURCHLAND FANO

# def compute_fano_churchland(
#     mv_dict: dict,
#     *,
#     n_bins: int = 10,
#     q_lo: float = 0.01,
#     q_hi: float = 0.99,
#     match_reps: int = 10,
#     weighting_epsilon: float = 0.01,
#     seed: int = 0,
# ) -> dict:
#     rng = np.random.default_rng(seed)

#     groups = {}
#     for (area, image, active) in mv_dict.keys():
#         groups.setdefault((area, active), []).append((area, image, active))

#     def _wls_slope_through_zero(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float, tuple[float, float]]:
#         Swxx = np.sum(w * x * x)
#         Swxy = np.sum(w * x * y)
#         B = Swxy / Swxx
#         resid = y - B * x
#         s2 = np.sum(w * resid * resid) / max(len(x) - 1, 1)
#         varB = s2 / Swxx
#         seB = float(np.sqrt(varB))
#         ci = (float(B - 2 * seB), float(B + 2 * seB))
#         return float(B), seB, ci

#     out = {}

#     for (area, active), keys in groups.items():
#         m_list = [np.asarray(mv_dict[k]["mean"], dtype=float) for k in keys]
#         v_list = [np.asarray(mv_dict[k]["variance"], dtype=float) for k in keys]

#         if "n_repeats" in mv_dict[keys[0]]:
#             n_list = [
#                 np.full(mv_dict[k]["mean"].shape[0], float(mv_dict[k]["n_repeats"]), dtype=float)
#                 for k in keys
#             ]
#         else:
#             n_list = [np.ones(m.shape[0], dtype=float) for m in m_list]

        

#         m_all = np.concatenate(m_list, axis=0) 
#         v_all = np.concatenate(v_list, axis=0) 
#         n_all = np.concatenate(n_list, axis=0) 
#         U, T = m_all.shape

#         lo = np.quantile(m_all, q_lo)
#         hi = np.quantile(m_all, q_hi)
#         edges = np.linspace(lo, hi, n_bins + 1)
#         bin_id = np.clip(np.digitize(m_all, edges, right=False) - 1, 0, n_bins - 1)

#         counts = np.zeros((n_bins, T), dtype=int)
#         for t in range(T):
#             bt = bin_id[:, t]
#             for b in range(n_bins):
#                 counts[b, t] = np.sum(bt == b)
#         target_count = counts.min(axis=1)

#         n_total = int(U)
#         n_total_matched = int(target_count.sum())

#         slope_all = np.empty(T, dtype=float)
#         reg_se_all = np.empty(T, dtype=float)
#         reg_ci_all = np.empty((T, 2), dtype=float)

#         for t in range(T):
#             x = m_all[:, t]
#             y = v_all[:, t]
#             w = n_all / (x + weighting_epsilon) ** 2
#             B, seB, ci = _wls_slope_through_zero(x, y, w)
#             slope_all[t] = B
#             reg_se_all[t] = seB
#             reg_ci_all[t, 0] = ci[0]
#             reg_ci_all[t, 1] = ci[1]

#         ff_point_all = v_all / (m_all + weighting_epsilon)

#         pop_sem_all = ff_point_all.std(axis=0, ddof=1) / np.sqrt(U)

#         slopes_reps = np.empty((match_reps, T), dtype=float)
#         keep_mask_rep0 = np.zeros((T, U), dtype=bool)

#         for rep in range(match_reps):
#             for t in range(T):
#                 bt = bin_id[:, t]
#                 to_keep = []

#                 for b in range(n_bins):
#                     k = target_count[b]
#                     if k <= 0:
#                         continue
#                     idx = np.flatnonzero(bt == b)
#                     sel = rng.choice(idx, size=k, replace=False)
#                     to_keep.append(sel)

#                 to_keep = np.concatenate(to_keep, axis=0)

#                 x = m_all[to_keep, t]
#                 y = v_all[to_keep, t]
#                 w = n_all[to_keep] / (x + weighting_epsilon) ** 2
#                 B, _, _ = _wls_slope_through_zero(x, y, w)
#                 slopes_reps[rep, t] = B

#                 if rep == 0:
#                     keep_mask_rep0[t, to_keep] = True

#         slope_mm = slopes_reps.mean(axis=0)
#         sem_mm_across_reps = slopes_reps.std(axis=0, ddof=1) / np.sqrt(match_reps)

#         mean_mm_rep0 = np.full((U, T), np.nan)
#         for t in range(T):
#             mask = keep_mask_rep0[t]
#             mean_mm_rep0[mask, t] = m_all[mask, t]

#         ff_point_mm_rep0 = np.full((U, T), np.nan, dtype=float)
#         for t in range(T):
#             mask = keep_mask_rep0[t]
#             ff_point_mm_rep0[mask, t] = v_all[mask, t] / (m_all[mask, t] + weighting_epsilon)

#         pop_sem_mm_rep0 = np.nanstd(ff_point_mm_rep0, axis=0, ddof=1) / np.sqrt(n_total_matched)

#         out[(area, active)] = {
#             "fano_raw": slope_all,
#             "unit_fano_raw": ff_point_all,
#             "sem_raw": pop_sem_all,            
#             "fano_mm": slope_mm,
#             "unit_fano_mm": ff_point_mm_rep0, 
#             "sem_mm": pop_sem_mm_rep0,
#             "sem_mm_across_reps": sem_mm_across_reps,
#             "fano_mm_across_reps": slopes_reps,
#             "target_count": target_count,
#             "counts": counts,
#             "n_total": n_total,
#             "n_total_matched": n_total_matched,
#         }

#     return out

def compute_fano_churchland(
    mv_dict: dict,
    *,
    n_bins: int = 10,
    q_lo: float = 0.01,
    q_hi: float = 0.99,
    match_reps: int = 10,
    weighting_epsilon: float = 0.01,
    seed: int = 0,
) -> dict:

    rng = np.random.default_rng(seed)

    groups = {}
    for (area, image, active) in mv_dict.keys():
        groups.setdefault((area, active), []).append((area, image, active))

    def _wls_slope_through_zero(
        x: np.ndarray, y: np.ndarray, w: np.ndarray
    ) -> tuple[float, float, tuple[float, float]]:
        Swxx = np.sum(w * x * x)
        Swxy = np.sum(w * x * y)
        B = Swxy / Swxx
        resid = y - B * x
        s2 = np.sum(w * resid * resid) / max(len(x) - 1, 1)
        varB = s2 / Swxx
        seB = float(np.sqrt(varB))
        ci = (float(B - 2 * seB), float(B + 2 * seB))
        return float(B), seB, ci

    out = {}

    for (area, active), keys in groups.items():
        m_list = [np.asarray(mv_dict[k]["mean"], dtype=float) for k in keys]
        v_list = [np.asarray(mv_dict[k]["variance"], dtype=float) for k in keys]

        if "n_repeats" in mv_dict[keys[0]]:
            n_list = [
                np.full(mv_dict[k]["mean"].shape[0], float(mv_dict[k]["n_repeats"]), dtype=float)
                for k in keys
            ]
        else:
            n_list = [np.ones(m.shape[0], dtype=float) for m in m_list]

        m_all = np.concatenate(m_list, axis=0)  # (U, T)
        v_all = np.concatenate(v_list, axis=0)  # (U, T)
        n_all = np.concatenate(n_list, axis=0)  # (U,)
        U, T = m_all.shape

        lo = np.quantile(m_all, q_lo)
        hi = np.quantile(m_all, q_hi)
        edges = np.linspace(lo, hi, n_bins + 1)
        bin_id = np.clip(np.digitize(m_all, edges, right=False) - 1, 0, n_bins - 1) 

        counts = np.zeros((n_bins, T), dtype=int)
        for t in range(T):
            bt = bin_id[:, t]
            for b in range(n_bins):
                counts[b, t] = np.sum(bt == b)
        target_count = counts.min(axis=1)  
        n_total = int(U)
        n_total_matched = int(target_count.sum()) 

        slope_all = np.empty(T, dtype=float)
        for t in range(T):
            x = m_all[:, t]
            y = v_all[:, t]
            w = n_all / (x + weighting_epsilon) ** 2
            B, _, _ = _wls_slope_through_zero(x, y, w)
            slope_all[t] = B

        ff_point_all = v_all / (m_all + weighting_epsilon)
        sem_raw = ff_point_all.std(axis=0, ddof=1) / np.sqrt(U)

        slopes_reps = np.full((match_reps, T), np.nan, dtype=float)

        ff_sum = np.zeros((U, T), dtype=float)
        ff_cnt = np.zeros((U, T), dtype=np.int32)

        for rep in range(match_reps):
            for t in range(T):
                bt = bin_id[:, t]
                to_keep_parts = []

                for b in range(n_bins):
                    k = int(target_count[b])
                    if k <= 0:
                        continue
                    idx = np.flatnonzero(bt == b)
                    if idx.size < k:
                        continue
                    sel = rng.choice(idx, size=k, replace=False)
                    to_keep_parts.append(sel)

                if len(to_keep_parts) == 0:
                    continue

                to_keep = np.concatenate(to_keep_parts, axis=0)

                x = m_all[to_keep, t]
                y = v_all[to_keep, t]
                w = n_all[to_keep] / (x + weighting_epsilon) ** 2
                B, _, _ = _wls_slope_through_zero(x, y, w)
                slopes_reps[rep, t] = B

                ff = v_all[to_keep, t] / (m_all[to_keep, t] + weighting_epsilon)
                ff_sum[to_keep, t] += ff
                ff_cnt[to_keep, t] += 1

        fano_mm = np.nanmean(slopes_reps, axis=0)  # (T,)
        sem_mm_across_reps = np.nanstd(slopes_reps, axis=0, ddof=1) / np.sqrt(max(match_reps, 1))

        unit_fano_mm = np.full((U, T), np.nan, dtype=float)
        mask = ff_cnt > 0
        unit_fano_mm[mask] = ff_sum[mask] / ff_cnt[mask]

        n_eff = np.sum(~np.isnan(unit_fano_mm), axis=0).astype(float)
        sem_mm = np.nanstd(unit_fano_mm, axis=0, ddof=1) / np.sqrt(np.maximum(n_eff, 1.0))

        out[(area, active)] = {
            "fano_raw": slope_all,
            "unit_fano_raw": ff_point_all,
            "sem_raw": sem_raw,
            "fano_mm": fano_mm,
            "unit_fano_mm": unit_fano_mm,
            "sem_mm": sem_mm,
            "sem_mm_across_reps": sem_mm_across_reps,
            "fano_mm_across_reps": slopes_reps,
            "target_count": target_count,
            "counts": counts,
            "n_total": n_total,
            "n_total_matched": n_total_matched,
        }

    return out

# 4: COMPUTING THE FANO METRICS

def compute_fano_metrics(
    fano_dict: dict,
    tpc_ts: np.ndarray
) -> pd.DataFrame:

    rows = []

    for key, dat in fano_dict.items():

        trace = dat["fano_mm"]

        windows = [[-0.25, 0], [0, 0.1], [0.1, 0.25]]
        window_indices = [np.searchsorted(tpc_ts, w) for w in windows]

        base  = trace[window_indices[0][0]:window_indices[0][1]].mean()
        early = trace[window_indices[1][0]:window_indices[1][1]].mean()
        late  = trace[window_indices[2][0]:window_indices[2][1]].mean()

        qi_early = (base - early) / (base + early)
        ri       = (late - early) / (late + early)

        trace_ew = trace[window_indices[1][0]:window_indices[2][1]]
        trace_min_idx = np.argmin(trace_ew)
        trace_min = trace_ew[trace_min_idx]

        evoked_time = tpc_ts[window_indices[1][0]:window_indices[2][1]]
        latency = evoked_time[trace_min_idx]

        onset_idx = window_indices[1][0]
        k = 5
        slope, intercept = np.polyfit(
            tpc_ts[onset_idx:onset_idx+k],
            trace[onset_idx:onset_idx+k],
            1
        )

        predicted_latency = (trace_min - intercept) / slope
        residual_latency  = latency - predicted_latency

        half_amplitude = trace_min + (trace[window_indices[1][0]] - trace_min) / 2
        above = trace_ew > half_amplitude
        crossings = np.diff(above.astype(int))

        up_idx   = np.where(crossings ==  1)[0] + 1
        down_idx = np.where(crossings == -1)[0] + 1

        if len(up_idx) > 0 and len(down_idx) > 0:
            fwhm = evoked_time[up_idx[0]] - evoked_time[down_idx[0]]
        else:
            fwhm = np.nan

        area, active = key

        rows.append({
            "area": area,
            "active": active,
            "base": base,
            "early": early,
            "late": late,
            "qi_early": qi_early,
            "ri": ri,
            "latency": latency,
            "init_slope": slope,
            "predicted_latency": predicted_latency,
            "residual_latency": residual_latency,
            "fwhm": fwhm
        })

    return pd.DataFrame(rows)

# 5: COMPILING SESSION SUMMARY

def build_summary(
      o_table: pd.DataFrame,
      u_table: pd.DataFrame,
      mv_dict: dict,
      areas: list
    ) -> dict:
   
    unit_counts = (
        u_table
        .loc[u_table["structure_acronym"].isin(areas), "structure_acronym"]
        .value_counts()
        .to_dict()
    ) 

    vsel_unit_counts = defaultdict(int)
    va_units = 0
    vp_units = 0 

    for key, dat in mv_dict.items():

        vsel_units = dat["mean"].shape[0]

        area, _, active = key

    if active:
      va_units += vsel_units
    else:
      vp_units += vsel_units

    vsel_unit_counts[(area, active)] += vsel_units

    return {
        "n_units": len(u_table),
        "n_units_per_area": unit_counts,
        "n_units_va": va_units,
        "n_units_vp": vp_units,
        "n_units_vsel_per_area": vsel_unit_counts,
        "n_onsets": len(o_table)
        }


# MAIN

def analyse_session(cfg: Config, s: PreprocessedSession) -> AnalysedSession:


    p = analysis_cache_paths(cfg, s.session_id)

    exists = {k: v.exists() for k, v in p.items()}
    true_keys = [k for k, ok in exists.items() if ok]
    false_keys = [k for k, ok in exists.items() if not ok]

    logger.info(f"[analysis] Existing cached files: {true_keys}")
    logger.info(f"[analysis] Missing cached files: {false_keys}")

    # LOADING

    if all(exists.values()):
        mv_dict = load_data(p["mv"])
        fano_dict = load_data(p["fano"])
        metrics = load_data(p["metrics"])
        tpc_ts = load_data(p["tpc_ts"])
        summary = load_data(p["summary_pkl"])

        return AnalysedSession(
            session_id=s.session_id,
            tpc_ts=tpc_ts,
            mv=mv_dict,
            fano=fano_dict,
            metrics=metrics,
            running=s.running,
            pupil=s.pupil,
            summary=summary,
        )
    
    # COMPUTING TIME-SAMPLED PSTH

    psth = s.psth
    tpc = s.tpc
    psth_ts, tpc_ts = time_sample_psth(cfg, psth, tpc)

    # COMPUTING MEAN AND VARAINCE

    areas = cfg.analysis.areas
    qc_ok = s.qc_ok

    o_table = s.o_table
    u_table = s.u_table
    v_table_thr = s.v_table_thr

    mv_dict = compute_mv(areas, psth_ts, o_table, u_table, v_table_thr, qc_ok)

    # COMPUTE CHURCHLAND FANO

    n_runs = cfg.analysis.n_runs
    fano_dict = compute_fano_churchland(mv_dict, match_reps=n_runs)

    # COMPUTING FANO METRICS

    metrics = compute_fano_metrics(fano_dict, tpc_ts)

    # COMPILING SUMMARY DATA FOR OUTPUT

    summary = build_summary(o_table, u_table, mv_dict, areas)

    # SAVING 

    if not p["mv"].exists():
        save_data(p["mv"], mv_dict)
        logger.info("[analysis] mv cached.")

    if not p["fano"].exists():
        save_data(p["fano"], fano_dict)
        logger.info("[analysis] fano cached.")

    if not p["metrics"].exists():
        save_data(p["metrics"], metrics)
        logger.info("[analysis] metrics cached.")

    if not p["tpc_ts"].exists():
        save_data(p["tpc_ts"], tpc_ts.astype(np.float32, copy=False))
        logger.info("[analysis] tpc_ts cached.")

    if not  p["summary_pkl"].exists():
        save_data(p["summary_pkl"], summary)
        logger.info("[analysis] summary cached (pickle).")

    return AnalysedSession(
        session_id=s.session_id,
        tpc_ts=tpc_ts,
        mv=mv_dict,
        fano=fano_dict,
        metrics=metrics,
        running=s.running,
        pupil=s.pupil,
        summary=summary,
    )

def get_all_mice(cfg: Config, all_sessions: dict):

  data = get_data(cfg.paths.dataset)
  metadata = data.get_ecephys_session_table()
  subset = metadata[metadata.index.isin(all_sessions.keys())]

  out = {}

  for midx, grp in subset.groupby("mouse_id"):
    mouse = {grp.loc[sid, 'experience_level']: all_sessions[sid] for sid in grp.index}
    out[midx] = mouse
    
  return out








  

  
    

  