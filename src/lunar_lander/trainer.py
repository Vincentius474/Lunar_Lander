from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np

from .agent import DQNAgent
from .environment import Action, EpisodeStatus, LanderEnv, LanderConfig
from .replay_buffer import ReplayBuffer

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


def validate_config(cfg: TrainConfig) -> None:
    """Reject invalid configurations before training begins."""

    positive_values = {
        "episodes": cfg.episodes,
        "max_steps": cfg.max_steps,
        "batch_size": cfg.batch_size,
        "buffer_size": cfg.buffer_size,
        "warmup_steps": cfg.warmup_steps,
        "train_every": cfg.train_every,
        "target_update": cfg.target_update,
        "log_every": cfg.log_every,
        "save_every": cfg.save_every,
        "hidden": cfg.hidden,
    }

    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero.")

    if cfg.buffer_size < cfg.batch_size:
        raise ValueError("buffer_size must be >= batch_size.")

    if cfg.warmup_steps < cfg.batch_size:
        raise ValueError("warmup_steps must be >= batch_size.")

    if cfg.warmup_steps > cfg.buffer_size:
        raise ValueError("warmup_steps cannot exceed buffer_size.")

    if cfg.lr <= 0:
        raise ValueError("lr must be greater than zero.")

    if not 0.0 <= cfg.gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1.")

    if not (
        0.0 <= cfg.epsilon_end <= cfg.epsilon_start <= 1.0
    ):
        raise ValueError(
            "Epsilon values must satisfy "
            "0 <= epsilon_end <= epsilon_start <= 1."
        )

    if not 0.0 < cfg.epsilon_decay <= 1.0:
        raise ValueError("epsilon_decay must be in (0, 1].")


def train(cfg: TrainConfig) -> dict[str, list]:
    """Train the DQN agent and return the recorded history."""

    validate_config(cfg)

    # Keep experiment results and checkpoints separate by seed.
    run_dir = Path(cfg.out_dir) / f"seed_{cfg.seed}"
    models_dir = Path(cfg.models_dir) / f"seed_{cfg.seed}"

    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    history_path = run_dir / "training_history.json"

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(cfg), file, indent=2)

    # Initialize the environment and infer its observation size.
    # env = LanderEnv(seed=cfg.env_seed)
    env_config = LanderConfig(potential_gamma=cfg.gamma)
    env = LanderEnv(config=env_config, seed=cfg.env_seed)
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
        seed=cfg.seed,
    )

    buffer = ReplayBuffer(
        capacity=cfg.buffer_size,
        obs_dim=obs_dim,
        seed=cfg.seed,
    )

    history: dict[str, list] = {
        "episode": [],
        "reward": [],
        "steps": [],
        "status": [],
        "truncated": [],
        "epsilon": [],
        "loss": [],
        "success": [],
    }

    recent_rewards: list[float] = []
    recent_success: list[int] = []

    global_step = 0
    start_time = time.time()

    def save_history() -> None:
        """Persist all metrics collected so far."""
        with history_path.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

    try:
        for ep in range(1, cfg.episodes + 1):
            # Reuse the seeded initial reset for episode 1.
            # Subsequent resets continue the environment RNG stream.
            if ep > 1:
                obs = env.reset()

            ep_reward = 0.0
            ep_loss_sum = 0.0
            ep_loss_count = 0
            info: dict = {}
            truncated = False

            for step in range(cfg.max_steps):
                # 1. Select an action.
                action = agent.select_action(obs)

                # 2. Advance the environment.
                next_obs, reward, done, info = env.step(action)

                # The trainer may end an episode when the step limit
                # is reached, even if the environment has not terminated.
                truncated = (
                    step == cfg.max_steps - 1 and not done
                )
                episode_finished = done or truncated

                # 3. Store the transition.
                # Mark truncations as finished so the next episode's
                # initial state is not treated as this transition's future.
                buffer.push(
                    obs,
                    action,
                    reward,
                    next_obs,
                    episode_finished,
                )

                obs = next_obs
                ep_reward += float(reward)
                global_step += 1

                # 4. Update the agent once enough data is available.
                if (
                    len(buffer) >= cfg.warmup_steps
                    and global_step % cfg.train_every == 0
                ):
                    batch = buffer.sample(cfg.batch_size)
                    loss = agent.update(batch)

                    ep_loss_sum += float(loss)
                    ep_loss_count += 1

                if episode_finished:
                    break

            # 5. Decay exploration after each episode.
            agent.end_episode()

            # 6. Record episode metrics.
            status = info.get("status", "unknown")

            if truncated and not done:
                status = "truncated"

            success = int(
                info.get("status") == EpisodeStatus.LANDED.value
            )

            recent_rewards.append(ep_reward)
            recent_success.append(success)

            if len(recent_rewards) > 100:
                recent_rewards.pop(0)
                recent_success.pop(0)

            history["episode"].append(ep)
            history["reward"].append(float(ep_reward))
            history["steps"].append(int(env.steps))
            history["status"].append(status)
            history["truncated"].append(bool(truncated))
            history["epsilon"].append(float(agent.epsilon))
            history["loss"].append(
                ep_loss_sum / ep_loss_count
                if ep_loss_count > 0
                else 0.0
            )
            history["success"].append(success)

            # 7. Log progress.
            if ep % cfg.log_every == 0 or ep == 1:
                avg_reward = float(np.mean(recent_rewards))
                avg_success = float(np.mean(recent_success))
                elapsed = time.time() - start_time

                print(
                    f"Episode {ep:5d} | "
                    f"Avg100 reward {avg_reward:8.2f} | "
                    f"Success {avg_success:6.2%} | "
                    f"Epsilon {agent.epsilon:.3f} | "
                    f"Steps {env.steps:4d} | "
                    f"Loss {history['loss'][-1]:.4f} | "
                    f"Time {elapsed:6.1f}s"
                )

            # 8. Save checkpoints and metrics together.
            if ep % cfg.save_every == 0:
                agent.save(str(models_dir / "dqn_latest.pt"))
                save_history()

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current progress...")

    finally:
        # Save the latest model and metrics even after interruption.
        agent.save(str(models_dir / "dqn_latest.pt"))
        save_history()

    # Save a separate final checkpoint only after all episodes finish.
    completed_all_episodes = len(history["episode"]) == cfg.episodes

    if completed_all_episodes:
        agent.save(str(models_dir / "dqn_final.pt"))

    elapsed = time.time() - start_time

    print(f"\nTraining stopped after {len(history['episode'])} episodes.")
    print(f"Elapsed time: {elapsed:.1f}s")
    print(f"Checkpoints: {models_dir}/")
    print(f"History: {history_path}")

    return history
