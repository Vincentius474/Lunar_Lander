# LunarLander AI — Phase 1

A standalone, CPU-friendly lunar landing simulation built with Python, Pygame, and NumPy. Phase 1 focuses on a deterministic, testable physics environment and a live visualization. The reinforcement-learning agent and analytics dashboard will be added in later phases.

## Features
- 2D lander with gravity, rotation, main-engine and side-engine thrust
- Fuel consumption and bounded world
- Landing pad, uneven lunar terrain, and collision detection
- Successful landing vs. crash conditions based on touchdown speed and angle
- Live HUD: position, velocity, angle, fuel, and episode status
- Keyboard controls and automatic reset
- Automated tests for basic physics and terminal states

## Requirements
- Python 3.10+
- Pygame 2.5+
- NumPy 1.24+

## Setup (Windows PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run the app using `.\.venv\Scripts\python.exe` directly.

## Run
```powershell
python -m src.lunar_lander
```

Controls:
- `A` / `D` or `←` / `→`: rotate
- `W` or `↑`: main engine
- `Space`: reset episode
- `Esc`: quit

The agent starts with a small initial horizontal velocity, so use short thrust bursts and counter-rotate before touchdown. The simulation uses simplified arcade physics, not a high-fidelity spacecraft model.

## Test
```powershell
python -m pytest
```

## Phase roadmap
1. **Phase 1 (this release):** physics, environment, visualization, tests.
2. Phase 2: DQN agent, replay buffer, target network, training checkpoints.
3. Phase 3: training dashboard, evaluation metrics, plots, and comparisons.
4. Phase 4: portfolio polish, reproducibility, screenshots/demo, and full documentation.

## Architecture
`LanderEnv` owns the simulation state and physics. `visualization.py` renders the state and collects human keyboard actions. The environment exposes a compact observation vector and discrete action interface so the same environment can later be controlled by a reinforcement-learning agent.
