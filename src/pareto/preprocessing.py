"""
preprocessing.py

Functions for preprocessing stimulus and unit data

Author: Elliot Pallister
"""


import numpy as np
import pandas as pd

# IMPORTING DATA FUNCTIONS FROM data_io.py
from pareto.data_io import get_cache, get_session, get_trials, get_unit_channels, get_units_by_area, get_spike_times, get_stimulus_presentations

"""

STIMULUS & TRIAL PROCESSING

""" 

# OBTAINS STIMULUS ONSETS FOR EACH TRIAL
def get_trial_stimulus_onsets(trials, image_onsets):

  start_index = np.searchsorted(image_onsets, trials['start_time'])
  stop_index = np.searchsorted(image_onsets, trials['stop_time'])

  onsets = [image_onsets[s:e] for s, e in zip(start_index, stop_index)]

  return onsets


# FILTERS TRIALS BY IMAGE
def get_image_trials(trials, image, image_onsets, non_abort=True):

  image_trials = trials[trials['initial_image_name'] == image].copy()
  trialwise_stimulus_onsets = get_trial_stimulus_onsets(image_trials, image_onsets)
  image_trials['stimulus_onsets'] = trialwise_stimulus_onsets

  if non_abort:
    return image_trials[image_trials['stimulus_onsets'].apply(len) >= 5]
  else:
    return image_trials
  
# RETURNS AN ARRAY OF STIMULUS TIMES FOR EACH TRIAL FRAME, 1-11
def arrange_image_onsets_to_trial(trials):

  events = (
    trials.assign(
      frame_index = lambda d: d['stimulus_onsets'].apply(lambda xs: list(range(1, len(xs) + 1)))
    )
    .explode(
      ['frame_index', 'stimulus_onsets']
    )[['frame_index', 'stimulus_onsets']]
  )
  
  return events

# GROUPS STIMULUS TIMES BY TRIAL FRAME, RETURNS SERIES
def group_stims_by_frame_index(events):

  grouped = (
    events
    .groupby(['frame_index'])['stimulus_onsets']
    .apply(list)
    .rename('onsets')
    .reset_index()
    .sort_values('frame_index', ignore_index=True)
  )

  return grouped

def trial_number_histogram(trials):

  events = arrange_image_onsets_to_trial(trials)
  counts = events['frame_index'].value_counts().sort_index()

  return counts

"""

RESPONSE PROCESSING

"""

# CONSTRUCTS PSTH MATRIX (O X T) FOR A GIVEN UNIT
def make_psth_matrix(spikes, onsets, windows, bin_size):

  window_dur = windows[0]+windows[3]

  bins = np.arange(0, window_dur+bin_size, bin_size)
  traces = []

  for i, start in enumerate(onsets):
    si = np.searchsorted(spikes, start-windows[0])
    ei = np.searchsorted(spikes, start+window_dur)
    trace = np.histogram(spikes[si:ei]-(start-windows[0]), bins)[0]
    traces.append(trace)

  return np.vstack(traces), bins

# CONSTRUCTS PSTH CUBE (U X O X T) FOR A SET OF UNITS
def make_psth_cube(units, spikes, onsets, windows, bin_size):

  cube = []
  unit_ids = []

  for uid, unit in units.iterrows():
    spike_times = spikes[uid]
    matrix, bins = make_psth_matrix(spike_times, onsets, windows, bin_size)
    cube.append(matrix)
    unit_ids.append(uid)

  return np.stack(cube, axis=0), unit_ids, bins - windows[0]


  
    

  





