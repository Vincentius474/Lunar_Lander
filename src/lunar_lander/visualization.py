from __future__ import annotations
import math
import sys
import pygame  # imported lazily inside run() for CI friendliness
from .environment import Action, EpisodeStatus, LanderEnv

"""Pygame renderer and keyboard controller for Phase 1."""

# Window / scale
WIDTH, HEIGHT = 1000, 700
SCALE_X, SCALE_Y = WIDTH / 800, HEIGHT / 600

# Palette
BG = (8, 13, 29)
WHITE = (235, 241, 255)
MUTED = (137, 154, 185)
CYAN = (72, 224, 235)
ORANGE = (255, 157, 69)
YELLOW = (255, 230, 140)
RED = (255, 91, 105)
GREEN = (110, 238, 165)
HULL_FILL = (26, 36, 56)

"""
    Lander geometry — ALL coordinates are in the lander's local frame.
    (0, 0) is the center of mass. +x right, +y down. The whole shape is
    rotated rigidly around (0, 0) each frame, so nothing can detach.
"""

# Octagonal hull, wider at the base (Apollo-LM silhouette)
LANDER_BODY: list[tuple[float, float]] = [
    (-10, -18),   # top-left
    (10, -18),    # top-right
    (16, -8),     # upper-right shoulder
    (16, 6),      # lower-right shoulder
    (12, 12),     # base-right
    (-12, 12),    # base-left
    (-16, 6),     # lower-left shoulder
    (-16, -8),    # upper-left shoulder
]

# Cockpit window (drawn on top of hull)
# a small circle centred here
LANDER_WINDOW = [(0, -11)]  

# Landing legs: each is a straight segment [p0, p1] in local space.
LANDER_LEGS: list[list[tuple[float, float]]] = [
    [(-12, 12), (-22, 28)],   # left leg strut
    [(12, 12), (22, 28)],     # right leg strut
]

# Foot pads: short horizontal segments at the bottom of each leg.
LANDER_FEET: list[list[tuple[float, float]]] = [
    [(-27, 28), (-17, 28)],   # left foot
    [(17, 28), (27, 28)],     # right foot
]

# Engine nozzle — trapezoid that attaches flush to the hull base (y = 12).
LANDER_NOZZLE: list[tuple[float, float]] = [
    (-5, 12),
    (5, 12),
    (4, 22),
    (-4, 22),
]

# Flame — triangle from the nozzle exit (y = 22) downward.
LANDER_FLAME_OUTER: list[tuple[float, float]] = [
    (-5, 22),
    (5, 22),
    (0, 40),
]
LANDER_FLAME_INNER: list[tuple[float, float]] = [
    (-2.5, 22),
    (2.5, 22),
    (0, 31),
]

# Rotation helper
def _rotate_and_translate(
    pts: list[tuple[float, float]], cx: float, cy: float,
    angle: float ) -> list[tuple[int, int]]:
    """
        Rotate local-space points by `angle` and translate to (cx, cy).
        Uses the same matrix as the physics (see physics.rotate_vector), so the
        rendered lander and the simulated lander agree on which way is 'up'.
    """
    ca, sa = math.cos(angle), math.sin(angle)
    out: list[tuple[int, int]] = []
    for lx, ly in pts:
        rx = lx * ca - ly * sa
        ry = lx * sa + ly * ca
        out.append((int(cx + rx), int(cy + ry)))
    return out

# Entry point
def run() -> None:
    try:
        # noqa: F811 (local import)
        import pygame  
    except ImportError as exc:
        raise SystemExit(
            "Pygame is required. Install it with: pip install pygame"
        ) from exc

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("LunarLander AI | Phase 1 Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    small = pygame.font.SysFont("consolas", 14)

    env = LanderEnv(seed=7)
    stars = [(int((i * 137) % WIDTH), int((i * 83) % 430)) for i in range(65)]

    running = True
    paused = False
    frame_action = Action.NOOP
    thrusting = False

    while running:
        clock.tick(60)
        thrusting = False
        frame_action = Action.NOOP

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    env.reset()
                    paused = False
                elif event.key == pygame.K_p:
                    paused = not paused

        # Keyboard: discrete action 
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            frame_action = Action.MAIN_ENGINE
            thrusting = True
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            frame_action = Action.ROTATE_LEFT
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            frame_action = Action.ROTATE_RIGHT
        elif keys[pygame.K_q]:
            frame_action = Action.LEFT_THRUSTER
        elif keys[pygame.K_e]:
            frame_action = Action.RIGHT_THRUSTER

        # Step physics 
        if not paused and env.status == EpisodeStatus.FLYING:
            env.step(frame_action)
            env.step(frame_action)  # two 1/30 s ticks per rendered frame

        # Render
        screen.fill(BG)
        for sx, sy in stars:
            pygame.draw.circle(screen, (76, 92, 125), (sx, sy), 1)

        # Terrain and landing pad
        terrain_pts = [
            (int(x * SCALE_X), int(y * SCALE_Y)) for x, y in env.terrain
        ]
        terrain_pts.extend([(WIDTH, HEIGHT), (0, HEIGHT)])
        pygame.draw.polygon(screen, (41, 50, 75), terrain_pts)

        # Pad line sampled from the terrain so it hugs the ground exactly
        pad_l = env.config.landing_pad_left
        pad_r = env.config.landing_pad_right
        pad_y_l = env._terrain_y(pad_l) * SCALE_Y
        pad_y_r = env._terrain_y(pad_r) * SCALE_Y
        pygame.draw.line(
            screen, CYAN,
            (int(pad_l * SCALE_X), int(pad_y_l)),
            (int(pad_r * SCALE_X), int(pad_y_r)),
            5,
        )

        # Lander
        cx = env.position[0] * SCALE_X
        cy = env.position[1] * SCALE_Y
        angle = env.angle

        if env.status == EpisodeStatus.LANDED:
            outline = GREEN
        elif env.status != EpisodeStatus.FLYING:
            outline = RED
        else:
            outline = WHITE

        # 1) Flame (behind everything)
        if thrusting and env.status == EpisodeStatus.FLYING and env.fuel > 0:
            outer = _rotate_and_translate(LANDER_FLAME_OUTER, cx, cy, angle)
            pygame.draw.polygon(screen, ORANGE, outer)
            inner = _rotate_and_translate(LANDER_FLAME_INNER, cx, cy, angle)
            pygame.draw.polygon(screen, YELLOW, inner)

        # 2) Legs and feet (behind hull)
        for seg in LANDER_LEGS:
            p = _rotate_and_translate(seg, cx, cy, angle)
            pygame.draw.line(screen, outline, p[0], p[1], 3)
        for seg in LANDER_FEET:
            p = _rotate_and_translate(seg, cx, cy, angle)
            pygame.draw.line(screen, outline, p[0], p[1], 3)

        # 3) Nozzle (behind hull so its top edge is hidden)
        nozzle = _rotate_and_translate(LANDER_NOZZLE, cx, cy, angle)
        pygame.draw.polygon(screen, HULL_FILL, nozzle)
        pygame.draw.polygon(screen, outline, nozzle, 2)

        # 4) Hull on top — hides every attachment seam
        hull = _rotate_and_translate(LANDER_BODY, cx, cy, angle)
        pygame.draw.polygon(screen, HULL_FILL, hull)
        pygame.draw.polygon(screen, outline, hull, 2)

        # 5) Cockpit window
        window_center = _rotate_and_translate(LANDER_WINDOW, cx, cy, angle)[0]
        pygame.draw.circle(screen, CYAN, window_center, 3)

        # HUD 
        pygame.draw.rect(screen, (16, 24, 44), (20, 18, 335, 142), border_radius=10)
        hud = [
            ("LUNAR LANDER / PHASE 1", CYAN),
            (f"POSITION  x={env.position[0]:6.1f}  y={env.position[1]:6.1f}", WHITE),
            (f"VELOCITY  vx={env.velocity[0]:5.1f} vy={env.velocity[1]:5.1f}", WHITE),
            (f"ANGLE     {math.degrees(env.angle):6.1f} deg", WHITE),
            (f"FUEL      {env.fuel:5.1f}%", ORANGE),
        ]

        for i, (label, col) in enumerate(hud):
            screen.blit(font.render(label, True, col), (34, 27 + i * 25))

        status_text = {
            EpisodeStatus.FLYING: "STATUS: DESCENDING — LAND ON THE CYAN PAD",
            EpisodeStatus.LANDED: "SUCCESSFUL LANDING! Press SPACE to retry",
            EpisodeStatus.CRASHED: "CRASHED — Press SPACE to retry",
            EpisodeStatus.OUT_OF_BOUNDS: "OUT OF BOUNDS — Press SPACE to retry",
        }[env.status]

        status_color = (
            GREEN
            if env.status == EpisodeStatus.LANDED
            else (RED if env.status != EpisodeStatus.FLYING else WHITE)
        )
        
        screen.blit(font.render(status_text, True, status_color), (24, HEIGHT - 72))
        screen.blit(
            small.render(
                "W/UP: MAIN ENGINE   A/D or LEFT/RIGHT: ROTATE   Q/E: SIDE THRUSTERS",
                True,
                MUTED,
            ),
            (24, HEIGHT - 43),
        )
        screen.blit(
            small.render("SPACE: RESET   P: PAUSE   ESC: QUIT", True, MUTED),
            (24, HEIGHT - 23),
        )

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    run()