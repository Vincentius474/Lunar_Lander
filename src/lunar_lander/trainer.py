from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np

from .agent import DQNAgent
from .environment import Action, EpisodeStatus, LanderEnv, LanderConfig
from .replay_buffer import ReplayBuffer

"""
Training loop for the DQN lander agent.
"""


@dataclass
class TrainConfig:
    # Environment
    env_seed: int = 0

    # Training loop
    episodes: int = 2500
    max_steps: int = 1000

    # Replay / updates
    batch_size: int = 64
    buffer_size: int = 50_000
    warmup_steps: int = 2000
    train_every: int = 1

    # Optimizer / RL
    lr: float = 1e-4
    gamma: float = 0.99
    hidden: int = 128

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.999

    # Target network
    target_update: int = 1000

    # Logging / saving
    seed: int = 0
    log_every: int = 20
    save_every: int = 100
    out_dir: str = "results"
    models_dir: str = "models"

def train(cfg: TrainConfig) -> dict:
    """
    Run a full training session. Returns the history dict.
    """
    # Output directories
    run_dir = Path(cfg.out_dir) / f"seed_{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(cfg.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save the config alongside the run so experiments are reproducible.
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Env / agent / buffer
    env = LanderEnv(seed=cfg.env_seed)
    obs = env.reset(seed=cfg.env_seed)
    obs_dim = obs.shape[0]
    n_actions = len(list(Action))

    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden=cfg.hidden,
        lr=cfg.lr,
        gamma=cfg.gamma,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay=cfg.epsilon_decay,
        target_update=cfg.target_update,
        seed=cfg.seed
    )
    buffer = ReplayBuffer(capacity=cfg.buffer_size, obs_dim=obs_dim, seed=cfg.seed)

    # History 
    history: dict[str, list] = {
        "episode": [],
        "reward": [],
        "steps": [],
        "status": [],
        "epsilon": [],
        "loss": [],
        "success": [],
    }
    recent_rewards: list[float] = []
    recent_success: list[int] = []

    global_step = 0
    t0 = time.time()

    # Main loop
    for ep in range(1, cfg.episodes + 1):
        obs = env.reset()
        ep_reward = 0.0
        ep_loss_sum = 0.0
        ep_loss_count = 0
        info: dict = {}

        for step in range(cfg.max_steps):
            # 1) Pick an action
            action = agent.select_action(obs)


            # The environment can terminate naturally, or the trainer
            # can truncate the episode at max_steps.
            next_obs, reward, done, info = env.step(action)

            truncated = (step == cfg.max_steps - 1) and not done
            episode_finished = done or truncated

            buffer.push(
                obs,
                action,
                reward,
                next_obs,
                episode_finished
            )

            obs = next_obs
            ep_reward += reward
            global_step += 1

            if len(buffer) >= cfg.warmup_steps:
                if global_step % cfg.train_every == 0:
                    batch = buffer.sample(cfg.batch_size)
                    ep_loss_sum += agent.update(batch)
                    ep_loss_count += 1

            if episode_finished:
                break

        
        # Decay exploration rate at the end of the episode
        agent.end_episode()

        # Bookkeeping 
        success = 1 if info.get("status") == EpisodeStatus.LANDED.value else 0
        recent_rewards.append(ep_reward)
        recent_success.append(success)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
            recent_success.pop(0)

        history["episode"].append(ep)
        history["reward"].append(ep_reward)
        history["steps"].append(env.steps)
        history["status"].append(info.get("status", "unknown"))
        history["epsilon"].append(agent.epsilon)
        history["loss"].append(ep_loss_sum / max(1, ep_loss_count))
        history["success"].append(success)

        # Periodic logging
        if ep % cfg.log_every == 0 or ep == 1:
            avg_r = float(np.mean(recent_rewards))
            avg_s = float(np.mean(recent_success))
            elapsed = time.time() - t0
            print(
                f"ep {ep:5d} | avg100 reward {avg_r:8.2f} | "
                f"success {avg_s:6.2%} | eps {agent.epsilon:.3f} | "
                f"steps {env.steps:4d} | {elapsed:6.1f}s"
            )

        # Periodic save
        if ep % cfg.save_every == 0:
            agent.save(str(models_dir / "dqn_latest.pt"))

    # Final save + history dump
    agent.save(str(models_dir / "dqn_final.pt"))
    with open(run_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete in {time.time() - t0:.1f}s")
    print(f"Checkpoints in {models_dir}/")
    print(f"History in {run_dir}/training_history.json")
    return history