from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
import numpy as np
from .physics import integrate

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

    # Keep this equal to the DQN agent's gamma.
    potential_gamma: float = 0.99
    potential_scale: float = 2.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("World width and height must be positive.")

        if self.dt <= 0:
            raise ValueError("dt must be positive.")

        if self.max_fuel <= 0:
            raise ValueError("max_fuel must be positive.")

        if not (
            0 <= self.landing_pad_left
            < self.landing_pad_right
            <= self.width
        ):
            raise ValueError(
                "Landing pad boundaries must be ordered "
                "and lie inside the world."
            )

        if not 0.0 <= self.potential_gamma <= 1.0:
            raise ValueError("potential_gamma must be between 0 and 1.")

        if self.potential_scale < 0:
            raise ValueError("potential_scale cannot be negative.")


class LanderEnv:
    """Lunar lander environment with a Gym-like reset/step API.

    Coordinates use a screen-style system: x increases to the right
    and y increases downward.
    """

    def __init__(
        self,
        config: LanderConfig | None = None,
        seed: int | None = None,
    ):
        self.config = config or LanderConfig()
        self.rng = np.random.default_rng(seed)
        self.terrain = self._make_terrain()
        self.reset()

    def _make_terrain(self) -> np.ndarray:
        """Create uneven terrain with an exactly flat landing pad."""
        c = self.config

        xs = np.unique(
            np.concatenate(
                (
                    np.linspace(0.0, c.width, 17),
                    np.array(
                        [c.landing_pad_left, c.landing_pad_right],
                        dtype=float,
                    ),
                )
            )
        )

        ys = c.terrain_height + self.rng.integers(
            -24, 25, size=len(xs)
        )

        pad_mask = (
            (xs >= c.landing_pad_left)
            & (xs <= c.landing_pad_right)
        )
        ys[pad_mask] = c.terrain_height

        ys[0] = c.terrain_height + 25
        ys[-1] = c.terrain_height + 25

        return np.column_stack((xs, ys)).astype(float)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Start a new episode. Supplying a seed resets the RNG."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.terrain = self._make_terrain()

        c = self.config

        self.position = np.array(
            [
                c.width * 0.5 + self.rng.uniform(-90.0, 90.0),
                90.0,
            ],
            dtype=float,
        )

        self.velocity = np.array(
            [
                self.rng.uniform(-13.0, 13.0),
                self.rng.uniform(-2.0, 5.0),
            ],
            dtype=float,
        )

        # Radians; zero means upright.
        self.angle = float(self.rng.uniform(-0.12, 0.12))
        self.angular_velocity = 0.0
        self.fuel = c.max_fuel
        self.status = EpisodeStatus.FLYING
        self.steps = 0
        self.total_reward = 0.0

        return self.observation()

    def observation(self) -> np.ndarray:
        """Return the nine normalized features expected by the DQN."""
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
                (
                    c.landing_pad_left + c.landing_pad_right
                ) / (2.0 * c.width),
            ],
            dtype=np.float32,
        )

    def step(
        self, action: int | Action
    ) -> tuple[np.ndarray, float, bool, dict]:
        """Apply one action and return observation, reward, done, info."""
        if self.status != EpisodeStatus.FLYING:
            return (
                self.observation(),
                0.0,
                True,
                {"status": self.status.value},
            )

        try:
            action = Action(int(action))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid lander action: {action!r}") from exc

        c = self.config
        self.steps += 1

        # Save the state before applying the action.
        previous_position = self.position.copy()
        previous_potential = self._potential()

        # Initialize reward before using +=.
        reward = -0.01

        acceleration = np.array([0.0, c.gravity], dtype=float)

        # Apply the chosen action.
        if action == Action.MAIN_ENGINE and self.fuel > 0:
            acceleration[0] += math.sin(self.angle) * c.main_thrust
            acceleration[1] -= math.cos(self.angle) * c.main_thrust
            self.fuel = max(
                0.0, self.fuel - c.fuel_main_rate * c.dt
            )
            reward -= 0.015

        elif action == Action.ROTATE_LEFT:
            self.angular_velocity -= c.rotation_speed * c.dt

        elif action == Action.ROTATE_RIGHT:
            self.angular_velocity += c.rotation_speed * c.dt

        elif action == Action.LEFT_THRUSTER and self.fuel > 0:
            acceleration[0] += c.side_thrust
            self.fuel = max(
                0.0, self.fuel - c.fuel_side_rate * c.dt
            )
            reward -= 0.005

        elif action == Action.RIGHT_THRUSTER and self.fuel > 0:
            acceleration[0] -= c.side_thrust
            self.fuel = max(
                0.0, self.fuel - c.fuel_side_rate * c.dt
            )
            reward -= 0.005

        # Update angular motion.
        self.angular_velocity *= 0.985
        self.angle += self.angular_velocity * c.dt
        self.angle = (
            (self.angle + math.pi) % (2.0 * math.pi)
        ) - math.pi

        # Integrate translational physics.
        self.position, self.velocity = integrate(
            self.position,
            self.velocity,
            acceleration,
            c.dt,
        )

        done = False

        # Check world boundaries before terrain contact so an
        # out-of-bounds transition cannot also be scored as a landing.
        out_of_bounds = (
            self.position[0] < 0.0
            or self.position[0] > c.width
            or self.position[1] < 20.0
        )

        if out_of_bounds:
            self.status = EpisodeStatus.OUT_OF_BOUNDS
            reward -= 150.0
            done = True

        else:
            # Detect a terrain crossing during this simulation step.
            collision_position = self._collision_point(
                previous_position, self.position
            )

            if collision_position is not None:
                self.position = collision_position

                on_pad = (
                    c.landing_pad_left + 8.0
                    <= self.position[0]
                    <= c.landing_pad_right - 8.0
                )

                safe_landing = (
                    on_pad
                    and abs(self.velocity[0])
                    <= c.landing_horizontal_speed_limit
                    and abs(self.velocity[1])
                    <= c.landing_speed_limit
                    and abs(self.angle) <= c.landing_angle_limit
                )

                self.status = (
                    EpisodeStatus.LANDED
                    if safe_landing
                    else EpisodeStatus.CRASHED
                )

                reward += 100.0 if safe_landing else -100.0
                done = True

        # Potential-based reward shaping.
        # For terminal transitions, use a terminal potential of zero.
        next_potential = 0.0 if done else self._potential()

        reward += c.potential_scale * (
            c.potential_gamma * next_potential
            - previous_potential
        )

        # Small additional penalties encourage alignment and slower
        # descent near the surface.
        if not done:
            pad_center = (
                c.landing_pad_left + c.landing_pad_right
            ) / 2.0

            dx = self.position[0] - pad_center
            altitude = c.terrain_height - self.position[1]

            if abs(dx) < 200.0:
                reward -= 0.0005 * abs(dx)

            if altitude < 150.0:
                reward -= 0.005 * abs(self.velocity[0])
                reward -= 0.003 * max(0.0, self.velocity[1])

        self.total_reward += reward

        return (
            self.observation(),
            float(reward),
            done,
            {
                "status": self.status.value,
                "fuel": self.fuel,
                "steps": self.steps,
                "total_reward": self.total_reward,
            },
        )

    def _terrain_y(self, x: float) -> float:
        """Interpolate terrain height at horizontal coordinate x."""
        return float(
            np.interp(x, self.terrain[:, 0], self.terrain[:, 1])
        )

    def _collision_point(
        self,
        previous_position: np.ndarray,
        current_position: np.ndarray,
    ) -> np.ndarray | None:
        """Find the first terrain contact along a movement segment.

        The lander center is considered in contact when it reaches
        terrain height minus 10 world units.
        """
        def clearance(position: np.ndarray) -> float:
            terrain_surface = self._terrain_y(float(position[0])) - 10.0
            return float(position[1] - terrain_surface)

        start_clearance = clearance(previous_position)
        end_clearance = clearance(current_position)

        if start_clearance >= 0.0:
            return previous_position.copy()

        if end_clearance < 0.0:
            return None

        # Binary search for the first contact along the segment.
        low = 0.0
        high = 1.0

        for _ in range(20):
            mid = (low + high) / 2.0
            candidate = (
                previous_position
                + mid * (current_position - previous_position)
            )

            if clearance(candidate) >= 0.0:
                high = mid
            else:
                low = mid

        return previous_position + high * (
            current_position - previous_position
        )

    def state_dict(self) -> dict:
        """Return a snapshot of the current physical state."""
        return {
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "angle": self.angle,
            "angular_velocity": self.angular_velocity,
            "fuel": self.fuel,
            "status": self.status.value,
            "steps": self.steps,
            "total_reward": self.total_reward,
        }

    def _potential(self) -> float:
        """Return a higher score for states more promising for landing."""
        c = self.config
        pad_center = (
            c.landing_pad_left + c.landing_pad_right
        ) / 2.0

        dx = (self.position[0] - pad_center) / c.width
        vx = self.velocity[0] / 60.0
        vy = max(0.0, self.velocity[1]) / 60.0
        angle_error = abs(self.angle) / math.pi
        angular_speed = abs(self.angular_velocity) / 4.0

        return (
            -1.5 * abs(dx)
            -0.5 * abs(vx)
            -0.8 * vy
            -0.5 * angle_error
            -0.2 * angular_speed
        )
