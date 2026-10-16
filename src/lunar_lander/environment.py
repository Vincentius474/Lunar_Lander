from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
import numpy as np
from .physics import integrate

"""
A minimal, deterministic lunar lander environment.
Coordinates are measured in world units: x grows rightward and y grows downward.
Actions are discrete to make the environment suitable for DQN in Phase 2.
"""

class Action(int, Enum):
    NOOP = 0
    MAIN_ENGINE = 1
    ROTATE_LEFT = 2
    ROTATE_RIGHT = 3
    LEFT_THRUSTER = 4
    RIGHT_THRUSTER = 5

class EpisodeStatus(str, Enum):
    FLYING = "flying"
    LANDED = "landed"
    CRASHED = "crashed"
    OUT_OF_BOUNDS = "out_of_bounds"

@dataclass(frozen=True)
class LanderConfig:
    width: float = 800.0
    height: float = 600.0
    gravity: float = 28.0
    main_thrust: float = 62.0
    side_thrust: float = 20.0
    rotation_speed: float = 1.8
    dt: float = 1.0 / 30.0
    max_fuel: float = 100.0
    fuel_main_rate: float = 15.0
    fuel_side_rate: float = 7.0
    landing_pad_left: float = 330.0
    landing_pad_right: float = 470.0
    landing_speed_limit: float = 30.0
    landing_horizontal_speed_limit: float = 22.0
    landing_angle_limit: float = 0.22
    terrain_height: float = 520.0

class LanderEnv:
    """
    Physics environment with Gym-like reset/step methods.
    """

    def __init__(self, config: LanderConfig | None = None, seed: int | None = None):
        self.config = config or LanderConfig()
        self.rng = np.random.default_rng(seed)
        self.terrain = self._make_terrain()
        self.reset()

    # Terrain
    # def _make_terrain(self) -> np.ndarray:
    #     """Stable, gently uneven terrain; flat section under the landing pad."""
    #     xs = np.linspace(0, self.config.width, 17)
    #     ys = self.config.terrain_height + self.rng.integers(-24, 25, size=len(xs))
    #     pad_mask = (xs >= self.config.landing_pad_left) & (xs <= self.config.landing_pad_right)
    #     ys[pad_mask] = self.config.terrain_height
    #     ys[0] = ys[-1] = self.config.terrain_height + 25
    #     return np.column_stack((xs, ys)).astype(float)
    
    def _make_terrain(self) -> np.ndarray:
        """Generate terrain with an exactly flat landing pad."""
        c = self.config

        # Include the pad boundaries as explicit terrain vertices.
        xs = np.unique(np.concatenate([
            np.linspace(0, c.width, 17),
            np.array([c.landing_pad_left, c.landing_pad_right]),
        ]))

        ys = c.terrain_height + self.rng.integers(
            -24, 25, size=len(xs)
        )

        # Keep the entire pad perfectly flat.
        pad_mask = (
            (xs >= c.landing_pad_left)
            & (xs <= c.landing_pad_right)
        )
        ys[pad_mask] = c.terrain_height

        # Keep the terrain edges slightly elevated.
        ys[0] = c.terrain_height + 25
        ys[-1] = c.terrain_height + 25

        return np.column_stack((xs, ys)).astype(float)

    # Gym-like API
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.terrain = self._make_terrain()    
        c = self.config

        self.position = np.array(
            [c.width * 0.5 + self.rng.uniform(-90, 90), 90.0], dtype=float
        )
        self.velocity = np.array(
            [self.rng.uniform(-13, 13), self.rng.uniform(-2, 5)], dtype=float
        )

         # radians | 0 = upright
        self.angle = float(self.rng.uniform(-0.12, 0.12)) 
        self.angular_velocity = 0.0
        self.fuel = c.max_fuel
        self.status = EpisodeStatus.FLYING
        self.steps = 0
        self.total_reward = 0.0

        return self.observation()

    def observation(self) -> np.ndarray:
        """Normalized state vector intended for a neural-network policy."""
        c = self.config
        return np.array(
            [
                self.position[0] / c.width,
                self.position[1] / c.height,
                self.velocity[0] / 60.0,
                self.velocity[1] / 60.0,
                math.sin(self.angle),
                math.cos(self.angle),
                self.angular_velocity / 3.0,
                self.fuel / c.max_fuel,
                (c.landing_pad_left + c.landing_pad_right) / (2 * c.width),
            ],
            dtype=np.float32,
        )

    def step(self, action: int | Action) -> tuple[np.ndarray, float, bool, dict]:
        if self.status != EpisodeStatus.FLYING:
            return self.observation(), 0.0, True, {"status": self.status.value}

        action = Action(int(action))
        c = self.config
        self.steps += 1
        acceleration = np.array([0.0, c.gravity], dtype=float)

        # Base step reward: a small time penalty to encourage efficiency
        # reward = -0.01

        # save previous state
        previous_position = self.position.copy()
        previous_velocity = self.velocity.copy()
        previous_angle = self.angle
        previous_angular_velocity = self.angular_velocity
        previous_potential = self._potential()

        reward += 2.0 * (
            self._potential() - previous_potential
        )

        if action == Action.MAIN_ENGINE and self.fuel > 0:
            acceleration[0] += math.sin(self.angle) * c.main_thrust
            acceleration[1] -= math.cos(self.angle) * c.main_thrust
            self.fuel = max(0.0, self.fuel - c.fuel_main_rate * c.dt)
            reward -= 0.015
        elif action == Action.ROTATE_LEFT:
            self.angular_velocity -= c.rotation_speed * c.dt
        elif action == Action.ROTATE_RIGHT:
            self.angular_velocity += c.rotation_speed * c.dt
        elif action == Action.LEFT_THRUSTER and self.fuel > 0:
            acceleration[0] += c.side_thrust
            self.fuel = max(0.0, self.fuel - c.fuel_side_rate * c.dt)
            reward -= 0.005
        elif action == Action.RIGHT_THRUSTER and self.fuel > 0:
            acceleration[0] -= c.side_thrust
            self.fuel = max(0.0, self.fuel - c.fuel_side_rate * c.dt)
            reward -= 0.005

        self.angular_velocity *= 0.985
        self.angle += self.angular_velocity * c.dt
        self.angle = (self.angle + math.pi) % (2 * math.pi) - math.pi
        self.position, self.velocity = integrate(
            self.position, self.velocity, 
            acceleration, c.dt
        )

        """
            Shaping (kept small so it never dominates the terminal reward)
            
            1. Horizontal alignment only applies within 200 units of the pad.
            2. Speed shaping only applies within 150 units of the ground.
            3. Coefficients reduced by roughly 4x.
            Worst-case shaping over an episode is now roughly -50, leaving the
            ±100 terminal reward as the dominant learning signal.
        """
        pad_center = (c.landing_pad_left + c.landing_pad_right) / 2
        dx = self.position[0] - pad_center
        altitude = c.terrain_height - self.position[1]

        if abs(dx) < 200.0:
            reward -= 0.0005 * abs(dx)

        if altitude < 150.0:
            reward -= 0.005 * abs(self.velocity[0])
            reward -= 0.003 * max(0.0, self.velocity[1])

        # Terminal conditions
        done = False
        terrain_y = self._terrain_y(self.position[0])

        if self.position[1] >= terrain_y - 10:
            on_pad = c.landing_pad_left + 8 <= self.position[0] <= c.landing_pad_right - 8
            safe = (
                on_pad
                and abs(self.velocity[0]) <= c.landing_horizontal_speed_limit
                and abs(self.velocity[1]) <= c.landing_speed_limit
                and abs(self.angle) <= c.landing_angle_limit
            )
            self.status = EpisodeStatus.LANDED if safe else EpisodeStatus.CRASHED
            reward += 100.0 if safe else -100.0
            done = True

        if (self.position[0] < 0 or self.position[0] > c.width or self.position[1] < 20):
            self.status = EpisodeStatus.OUT_OF_BOUNDS
            reward -= 150.0
            done = True

        self.total_reward += reward
        return (
            self.observation(),
            float(reward),
            done,
            {
                "status": self.status.value,
                "fuel": self.fuel,
                "steps": self.steps,
                "total_reward": self.total_reward
            }
        )

    # Helpers
    def _terrain_y(self, x: float) -> float:
        return float(np.interp(x, self.terrain[:, 0], self.terrain[:, 1]))

    def state_dict(self) -> dict:
        return {
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "angle": self.angle,
            "angular_velocity": self.angular_velocity,
            "fuel": self.fuel,
            "status": self.status.value,
            "steps": self.steps,
            "total_reward": self.total_reward
        }

    def _potential(self) -> float:
        """Higher potential means a more promising landing state."""
        c = self.config
        pad_center = (
            c.landing_pad_left + c.landing_pad_right
        ) / 2

        dx = (self.position[0] - pad_center) / c.width
        vx = self.velocity[0] / 60.0
        vy = max(0.0, self.velocity[1]) / 60.0

        # Angle is already wrapped to [-pi, pi].
        angle_error = abs(self.angle) / math.pi
        angular_speed = abs(self.angular_velocity) / 4.0

        return (
            -1.5 * abs(dx)
            -0.5 * abs(vx)
            -0.8 * vy
            -0.5 * angle_error
            -0.2 * angular_speed
        )
