from __future__ import annotations

import argparse
import math
import sys
import pygame

from lunar_lander.agent import DQNAgent
from lunar_lander.environment import Action, EpisodeStatus, LanderEnv
from lunar_lander.visualization import (
    BG, CYAN, GREEN, HEIGHT,
    HULL_FILL, LANDER_BODY, LANDER_FEET,
    LANDER_FLAME_INNER, LANDER_FLAME_OUTER, LANDER_LEGS,
    LANDER_NOZZLE, LANDER_WINDOW, MUTED, ORANGE, RED, SCALE_X,
    SCALE_Y, WHITE, WIDTH, YELLOW, _rotate_and_translate)

"""
Play a trained DQN agent in the Pygame window.

Run with: python -m scripts.play --model models/dqn_final.pt
"""

def _draw_lander(screen, env, thrusting: bool) -> None:
    """
    Same drawing as visualization.run(), factored out for reuse.
    """
    cx = env.position[0] * SCALE_X
    cy = env.position[1] * SCALE_Y
    angle = env.angle

    if env.status == EpisodeStatus.LANDED:
        outline = GREEN
    elif env.status != EpisodeStatus.FLYING:
        outline = RED
    else:
        outline = WHITE

    if thrusting and env.status == EpisodeStatus.FLYING and env.fuel > 0:
        outer = _rotate_and_translate(LANDER_FLAME_OUTER, cx, cy, angle)
        pygame.draw.polygon(screen, ORANGE, outer)
        inner = _rotate_and_translate(LANDER_FLAME_INNER, cx, cy, angle)
        pygame.draw.polygon(screen, YELLOW, inner)

    for seg in LANDER_LEGS:
        p = _rotate_and_translate(seg, cx, cy, angle)
        pygame.draw.line(screen, outline, p[0], p[1], 3)
    for seg in LANDER_FEET:
        p = _rotate_and_translate(seg, cx, cy, angle)
        pygame.draw.line(screen, outline, p[0], p[1], 3)

    nozzle = _rotate_and_translate(LANDER_NOZZLE, cx, cy, angle)
    pygame.draw.polygon(screen, HULL_FILL, nozzle)
    pygame.draw.polygon(screen, outline, nozzle, 2)

    hull = _rotate_and_translate(LANDER_BODY, cx, cy, angle)
    pygame.draw.polygon(screen, HULL_FILL, hull)
    pygame.draw.polygon(screen, outline, hull, 2)

    wc = _rotate_and_translate(LANDER_WINDOW, cx, cy, angle)[0]
    pygame.draw.circle(screen, CYAN, wc, 3)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="models/dqn_final.pt")
    p.add_argument("--episodes", type=int, default=5,
                   help="Number of episodes to watch before exiting.")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("LunarLander AI | Trained Agent")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    small = pygame.font.SysFont("consolas", 14)

    env = LanderEnv(seed=args.seed)
    obs = env.reset(seed=args.seed)

    # Build an agent matching the checkpoint's dims.
    agent = DQNAgent(obs_dim=obs.shape[0], n_actions=len(list(Action)), seed=args.seed)
    agent.load(args.model)
    agent.epsilon = 0.0  # pure exploitation

    stars = [(int((i * 137) % WIDTH), int((i * 83) % 430)) for i in range(65)]
    episode = 1
    episodes_done: list[str] = []
    paused = False
    running = True

    while running:
        clock.tick(60)
        thrusting = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_SPACE:
                    obs = env.reset()
                    episode += 1

        # ---- Advance simulation ----
        if not paused and env.status == EpisodeStatus.FLYING:
            action = agent.select_action(obs, greedy=True)
            obs, _, done, info = env.step(action)
            obs, _, done, info = env.step(action)  # 2 ticks per frame like Phase 1
            thrusting = action == int(Action.MAIN_ENGINE)

        # ---- Render ----
        screen.fill(BG)
        for sx, sy in stars:
            pygame.draw.circle(screen, (76, 92, 125), (sx, sy), 1)

        terrain_pts = [(int(x * SCALE_X), int(y * SCALE_Y)) for x, y in env.terrain]
        terrain_pts.extend([(WIDTH, HEIGHT), (0, HEIGHT)])
        pygame.draw.polygon(screen, (41, 50, 75), terrain_pts)

        pad_l = env.config.landing_pad_left
        pad_r = env.config.landing_pad_right
        pygame.draw.line(
            screen, CYAN,
            (int(pad_l * SCALE_X), int(env._terrain_y(pad_l) * SCALE_Y)),
            (int(pad_r * SCALE_X), int(env._terrain_y(pad_r) * SCALE_Y)),
            5,
        )

        _draw_lander(screen, env, thrusting)

        # ---- HUD ----
        pygame.draw.rect(screen, (16, 24, 44), (20, 18, 335, 142), border_radius=10)
        hud = [
            ("LUNAR LANDER / TRAINED AGENT", CYAN),
            (f"EPISODE   {episode} / {args.episodes}", WHITE),
            (f"VELOCITY  vx={env.velocity[0]:5.1f} vy={env.velocity[1]:5.1f}", WHITE),
            (f"ANGLE     {math.degrees(env.angle):6.1f} deg", WHITE),
            (f"FUEL      {env.fuel:5.1f}%", ORANGE),
        ]
        for i, (label, col) in enumerate(hud):
            screen.blit(font.render(label, True, col), (34, 27 + i * 25))

        if env.status != EpisodeStatus.FLYING:
            text = {
                EpisodeStatus.LANDED: "LANDED  —  SPACE for next episode",
                EpisodeStatus.CRASHED: "CRASHED —  SPACE for next episode",
                EpisodeStatus.OUT_OF_BOUNDS: "OUT OF BOUNDS —  SPACE for next",
            }[env.status]
            color = GREEN if env.status == EpisodeStatus.LANDED else RED
            screen.blit(font.render(text, True, color), (24, HEIGHT - 72))
            if env.status.value not in episodes_done:
                episodes_done.append(env.status.value)

        screen.blit(small.render("SPACE: NEXT EPISODE   P: PAUSE   ESC: QUIT", True, MUTED),
                    (24, HEIGHT - 43))
        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()