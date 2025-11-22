"""
stats.py

Functions for performing statistical analysis on preprocessed data

Author: Elliot Pallister
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata
from statsmodels.stats.multitest import fdrcorrection


"""

UNIT FILTERING


"""

def avg_response(spike_times, image_onsets, window):
  t0, t1 = window
  rates = []
  for start in image_onsets:
    n_spikes = np.sum((spike_times >= start + t0) & (spike_times < start + t1))
    rates.append(n_spikes / (t1 - t0))
  
  return np.array(rates)

def wilcoxon_one_sided_paired(stim, base, alternative='greater'):

  stim = np.asarray(stim, dtype=float)
  base = np.asarray(base, dtype=float)
  
  D = stim - base
  nz = D != 0

  if not np.any(nz):
      return dict(p=1.0, mode='all_zero', n_eff=0, median_diff=0.0, r_rb=0.0, dz=0.0)
  
  Dn = D[nz]

  try:
    stat, p = wilcoxon(Dn, zero_method='wilcox', alternative=alternative,
                        correction=False, mode='exact')
    mode = 'exact'
  except ValueError:
    stat, p = wilcoxon(Dn, zero_method='wilcox', alternative=alternative,
                        correction=True, mode='approx')
    mode = 'approx'

  abs_ranks = rankdata(np.abs(Dn), method='average')
  W_plus  = (abs_ranks * (Dn > 0)).sum()
  W_minus = (abs_ranks * (Dn < 0)).sum()
  r_rb = (W_plus - W_minus) / (W_plus + W_minus)

  return dict(
        p=float(p), mode=mode, n_eff=int(Dn.size),
        median_diff=float(np.median(Dn)), r_rb=float(r_rb)
    )

def visual_selectivity_filter(
    units, 
    spike_times, 
    onsets,
    baseline_window,
    evoked_window,
    alpha=0.05,
    effect_threshold=0.25,
    effect_metric='r_rb',
    alternative='greater'
):

  rows = []

  for uid, unit in units.iterrows():
    
    spikes = spike_times[uid]

    base = avg_response(spikes, onsets, baseline_window)
    stim = avg_response(spikes, onsets, evoked_window)

    metrics = wilcoxon_one_sided_paired(stim, base, alternative)
    rows.append(dict(index=uid, **metrics))

  stats = pd.DataFrame(rows).set_index('index').reindex(units.index)

  rejected, p_fdr = fdrcorrection(stats['p'].fillna(1.0).values, alpha=alpha)
  stats['p_fdr'] = p_fdr
  stats['reject'] = rejected

  eff = stats[effect_metric].fillna(0.0).values
  stats['pass_effect'] = np.abs(eff) > effect_threshold

  mask = (stats['reject'] & stats['pass_effect']).values

  return mask, stats

"""

RESPONSE-LEVEL ANALYSIS

"""

def roc_analysis(dist_1: np.ndarray, dist_2: np.ndarray):

  conc_ds = np.concatenate((dist_1, dist_2))
  max_value = np.max(conc_ds)
  min_value = np.min(conc_ds)
  thetas = np.linspace(min_value, max_value, 1000)

  TPR = []
  FPR = []

  for threshold in thetas:
    TPR.append(np.sum(dist_2 >= threshold) / len(dist_2))
    FPR.append(np.sum(dist_1 >= threshold) / len(dist_1))

  TPR, FPR = np.array(TPR), np.array(FPR)

  return TPR, FPR

def get_snr(
  responses,
  baseline_window, 
  evoked_window,
  bins
  ):

  bi = np.searchsorted(bins, baseline_window)
  ei = np.searchsorted(bins, evoked_window)

  baseline = responses[:, bi[0]:bi[1]].mean(axis=1)
  evoked = responses[:, ei[0]:ei[1]].mean(axis=1)

  bsub = evoked - baseline
  signal = bsub.mean()
  noise = np.std(bsub)

  snr = signal / noise

  return snr

def auc_roc(responses, chunk_size):

  n_chunks = int(round(responses.shape[1]/chunk_size))

  unit_chunks = []

  for unit in responses:

    chunks = []

    for i in range(n_chunks):
      chunk = unit[i*20:(i+1)*20]
      chunks.append(chunk)
  
    chunks = np.array(chunks)
    unit_chunks.append(chunks)

  unit_chunks = np.array(unit_chunks)

  unit_TPRs, unit_FPRs, unit_AUCs = [], [], []

  for unit in unit_chunks:

    TPRs, FPRs, AUCs = [], [], []

    for c in range(n_chunks-1):

      TPR, FPR = roc_analysis(unit[0], unit[c+1])
      TPR, FPR = np.array(TPR), np.array(FPR)
      TPRs.append(TPR)
      FPRs.append(FPR)
      order = np.argsort(FPR)
      auc = np.trapz(TPR[order], FPR[order]) 
      if auc < 0.5:
        auc = 1 - auc
      AUCs.append(auc)

    TPRs, FPRs, AUCs = np.array(TPRs), np.array(FPRs), np.array(AUCs)
    unit_TPRs.append(TPRs)
    unit_FPRs.append(FPRs)
    unit_AUCs.append(AUCs)

  unit_TPRs, unit_FPRs, unit_AUCs = np.array(unit_TPRs), np.array(unit_FPRs), np.array(unit_AUCs)

  return unit_TPRs, unit_FPRs, unit_AUCs

def population_sparseness(
    responses,
    baseline_window,
    evoked_window,
    bins
    ):
  
  bi = np.searchsorted(bins, baseline_window)
  ei = np.searchsorted(bins, evoked_window)

  baseline = responses[:, :, bi[0]:bi[1]].mean(axis=1)
  evoked = responses[:, :, ei[0]:ei[1]].mean(axis=1)

  bsub = evoked - baseline

  mu_evoked = bsub.mean(axis=1)

  print(mu_evoked.shape)

  num = (1 - ((mu_evoked.mean() ** 2)/((mu_evoked ** 2).mean())))
  den = (1 - (1 / mu_evoked.shape[0]))

  sparsity = num / den

  return sparsity