from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fano_project.config import Config, load_config
from fano_project.preprocessing.preprocessing_logic import preprocess_session
from fano_project.analysis.analysis_logic import analyse_session

def parse_args() -> argparse.Namespace:

  p = argparse.ArgumentParser(description="Run preprocessing and cache outputs")
  p.add_argument(
    "--config",
    type=str,
    default=None,
    help="Path to config YAML. If omitted, uses PROJECT CODE/config/default.yaml"
  )

  p.add_argument(
    "--session-id",
    type=int,
    default=None,
    help="Run a single session id (overrides config list)."
  )

  p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="DEBUG, INFO, WARNING, ERROR",
    )

  return p.parse_args()

def main() -> None:

  args = parse_args()

  logging.basicConfig(
    level=getattr(logging, args.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
  )

  logger = logging.getLogger("run_preprocessing")

  config_path = Path(args.config)

  cfg = load_config(config_path)

  if args.session_id is not None:
        session_ids = [int(args.session_id)]
  else:
      session_ids = cfg.raw.get("dataset", {}).get("session_ids", [])
      if not session_ids:
          raise ValueError(
              "No session IDs provided. Either pass --session-id or set dataset.session_ids in the config."
          )
      
  logger.info(f"Using config: {config_path}")
  logger.info(f"Project root: {cfg.root}")
  logger.info(f"Dataset path: {cfg.paths.dataset}")
  logger.info(f"Cache path: {cfg.paths.cache}")
  logger.info(f"Running {len(session_ids)} session(s)")

  ans_dict = {}

  for sid in session_ids:
        
        sid = int(sid)
        logger.info(f"--- Session {sid} ---")

        ps = preprocess_session(cfg, session_id=sid)
        logger.info(f"Session {sid} preprocessing complete, beginning analysis...")

        ans = analyse_session(cfg, ps)
        logger.info(f"Session {sid} analysis complete")

        ans_dict[sid] = ans
        

if __name__ == "__main__":
    main()