"""
data_io.py

Functions for loading sessions, units, spikes and stimulus metadata

Author: Elliot Pallister
"""

from pathlib import Path
import numpy as np
import pandas as pd
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache

PROJECT_ROOT = Path(__file__).resolve().parents[2 ]
DATA_DIR = PROJECT_ROOT / 'data'
CACHE_DIR = DATA_DIR / 'allen_cache'

# LOADING CACHE

def get_cache(cache_dir: Path | None = None):

  cache_dir = cache_dir or CACHE_DIR
  cache_dir.mkdir(parents=True, exist_ok=True)
  cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=cache_dir)
  cache.load_latest_manifest()

  return cache

# LOADING SESSION DATA

def get_session(session_id: int, cache=None):

  if cache == None:
    cache = get_cache()

  return cache.get_ecephys_session(session_id)

# LOADING UNITS FROM SESSION

def get_unit_channels(session) -> pd.DataFrame:

  units = session.get_units().copy()
  channels = session.get_channels().copy()

  unit_channels = units.merge(channels, left_on='peak_channel_id', right_index=True)
  unit_channels = unit_channels.sort_values('probe_vertical_position', ascending=False)

  return unit_channels.copy()

# LOADING SPIKE TIMES

def get_spike_times(session) -> dict[int, np.ndarray]:

  spikes = {}
  units = get_unit_channels(session)

  for uid in units.index:
    spikes[uid] = session.spike_times[uid]
  
  return spikes

# LOADING STIMULUS PRESENTATIONS

def get_stimulus_presentations(session) -> pd.DataFrame:

  stim_df = session.stimulus_presentations.copy()
  stim_df = stim_df[stim_df['stimulus_name'].str.startswith('Natural_Image')]

  return stim_df

def get_trials(session) -> pd.DataFrame:

  trials = session.trials
  # later add in filtering logic based on hits, misses, false_alarms, correct_rejects, aborts

  return trials

# LOADING UNITS BY AREA

def get_units_by_area(units: pd.DataFrame, areas: list[str] | None = None):

  if areas == None:
    return units
  
  return units.loc[units['structure_acronym'].isin(areas)].copy()


# SUMMARISING SESSION DATA

def session_summary(session_id = int, cache=None) -> dict:

  session = get_session(session_id, cache)
  unit_channels = get_unit_channels(session)
  trials = get_trials(session)
  stims = get_stimulus_presentations(session)

  return {
    "session_id": session_id,
    "n_units": len(unit_channels),
    "n_trials": len(trials),
    "n_stims": len(stims),
    "areas": sorted(unit_channels['structure_acronym'].unique().tolist())
  }

# TESTING

if __name__ == "__main__":

  print("Project root:", PROJECT_ROOT)
  print("Data directory:", DATA_DIR)
  print("Cache directory:", CACHE_DIR)

  cache = get_cache()
  test_session = 1044385384
  session = get_session(test_session, cache)

  unit_channels = get_unit_channels(session)
  trials = get_trials(session)
  stims = get_stimulus_presentations(session)

  areas_of_interest = ['VISp']
  area_units = get_units_by_area(unit_channels, areas_of_interest)

  print(session_summary(test_session, cache))
  print(f"Area of interest summary {areas_of_interest}: {len(area_units)} units")










  
