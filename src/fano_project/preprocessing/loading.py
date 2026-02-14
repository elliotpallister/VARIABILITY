import numpy as np
import pandas as pd

from pathlib import Path
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache

from fano_project.config import Config

import logging

logger = logging.getLogger(__name__)

# LOADING DATA MANIFEST FROM ALLENSDK

def get_data(data_dir: Path) -> VisualBehaviorNeuropixelsProjectCache:

  logger.info("Initialising AllenSDK cache")

  data_dir.mkdir(parents=True, exist_ok=True)
  data = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=data_dir)

  logger.info("Loading latest manifest")
  data.load_latest_manifest()

  logger.info("AllenSDK cache ready")
  return data

# LOADING SESSION SPECIFIC DATA FROM ALLENSDK

def load_session(cfg: Config, session_id: int):
  logger.info("Loading session %d", session_id)

  data = get_data(cfg.paths.dataset)

  session = data.get_ecephys_session(session_id)
  logger.info("Session loaded")
  print("session loaded", flush=True)

  units = session.get_units().copy()
  channels = session.get_channels().copy()

  u_table = units.merge(channels, left_on='peak_channel_id', right_index=True)
  u_table = u_table.sort_values('probe_vertical_position', ascending=False)
  logger.info("Loaded %d units", len(u_table))
  print("units loaded", flush=True)

  spikes = {}
  for uid in u_table.index:
    spikes[uid] = session.spike_times[uid]
  logger.info("Extracted spike times for %d units", len(spikes))
  print("spikes done", flush=True)

  s_table = session.stimulus_presentations.copy()
  s_table = s_table[s_table['stimulus_name'].str.startswith('Natural_Image')]
  print("s_table done", flush=True)

  t_table = session.trials.copy()
  logger.info("Loaded %d stimulus presentations, %d trials", len(s_table), len(t_table))

  print("where I get up to")

  running = session.running_speed
  pupil = session.eye_tracking
  licks = session.licks

  return {
    "u_table": u_table,
    "s_table": s_table,
    "t_table": t_table,
    "spikes": spikes,
    "running": running,
    "pupil": pupil,
    "licks": licks
  }


  



    