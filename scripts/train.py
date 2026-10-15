from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Make the project root importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_lander.trainer import TrainConfig, train

"""
Train a DQN agent. 
Run with: python -m scripts.train
"""

def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as f:
        raw = json.load(f)
    # Flatten: the JSON has {"environment": {...}, "training": {...}} 
    # TrainConfig only cares about the "training" section plus env_seed.
    return raw.get("training", {})


def main() -> None:
    p = argparse.ArgumentParser(description="Train a DQN lunar lander agent.")
    p.add_argument("--episodes", type=int, default=None,
                   help="Number of training episodes.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed.")
    p.add_argument("--config", type=str, default=None,
                   help="Path to a JSON config (e.g. configs/default.json).")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Where to write training history.")
    p.add_argument("--models-dir", type=str, default=None,
                   help="Where to save checkpoints.")
    args = p.parse_args()

    # Start from config file (if any), then apply CLI overrides.
    kwargs = _load_config(args.config)
    if args.episodes is not None:
        kwargs["episodes"] = args.episodes
    if args.seed is not None:
        kwargs["seed"] = args.seed
        kwargs["env_seed"] = args.seed
    if args.out_dir is not None:
        kwargs["out_dir"] = args.out_dir
    if args.models_dir is not None:
        kwargs["models_dir"] = args.models_dir

    cfg = TrainConfig(**kwargs)
    train(cfg)


if __name__ == "__main__":
    main()