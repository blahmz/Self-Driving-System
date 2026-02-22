"""
Road & Traffic Management
Oncoming traffic uses a metered batch system:
  - One car at a time in the oncoming lane
  - After each car passes, a long mandatory quiet window opens
  - This guarantees the ego car always gets clear windows to overtake
"""
import pygame
import random
import os
from config import (
    WIDTH, HEIGHT, ROAD_WIDTH, LANE_WIDTH,
    ROAD_LEFT, ROAD_RIGHT, LANE_CENTERS,
    WHITE, YELLOW, ASPHALT, GRASS_COL,
    VEHICLE_SCALE, CAR_WIDTH, _CAR_H_REF,
    SAME_DIR_NUM_CARS, SAME_DIR_SPEED_MIN, SAME_DIR_SPEED_MAX,
    SAME_DIR_MIN_GAP, SAME_DIR_SPAWN_GAP, SAME_DIR_SPAWN_PROB,
    ONCOMING_NUM_CARS, ONCOMING_SPEED_MIN, ONCOMING_SPEED_MAX,
    ONCOMING_MIN_GAP, ONCOMING_SPAWN_GAP, ONCOMING_SPAWN_PROB,
    POST_OVERTAKE_SPAWN_COOLDOWN_FRAMES,
    MIN_SPAWN_DIST_AHEAD, POST_OVERTAKE_SPAWN_BUFFER,
)

def load_image(name, fallback_color=(120, 120, 120), fallback_size=(50, 88)):
    for path in [name, name.replace('.jpg', '.png'), name.replace('.png', '.jpg')]:
        if os.path.exists(path):
            return pygame.image.load(path).convert_alpha()
    surf = pygame.Surface(fallback_size, pygame.SRCALPHA)
    surf.fill(fallback_color)
    return surf


class Road:
    TOP    = -1_000_000
    BOTTOM =  1_000_000

    def __init__(self):
        self.left_x  = ROAD_LEFT
        self.right_x = ROAD_RIGHT
        self.borders = [
            [{'x': self.left_x,  'y': self.TOP}, {'x': self.left_x,  'y': self.BOTTOM}],
            [{'x': self.right_x, 'y': self.TOP}, {'x': self.right_x, 'y': self.BOTTOM}],
        ]
        self.lane_positions = LANE_CENTERS
        raw_grass = load_image('grass.jpg',        GRASS_COL, (WIDTH, HEIGHT))
        raw_road  = load_image('road_texture.jpg', ASPHALT,   (ROAD_WIDTH, HEIGHT))
        self.grass_surf = pygame.transform.scale(raw_grass, (WIDTH, HEIGHT))
        self.road_surf  = pygame.transform.scale(raw_road,  (ROAD_WIDTH, HEIGHT))

    def draw(self, screen, off1, off2):
        for dy in range(-1, 3):
            screen.blit(self.grass_surf, (0, int(off1) + dy * HEIGHT))
            screen.blit(self.grass_surf, (0, int(off2) + dy * HEIGHT))

        for dy in range(-1, 3):
            screen.blit(self.road_surf, (self.left_x, int(off1) + dy * HEIGHT))
            screen.blit(self.road_surf, (self.left_x, int(off2) + dy * HEIGHT))

        cx = self.left_x + LANE_WIDTH
        for off in (off1, off2):
            for dy in range(-1, 3):
                y0 = int(off) + dy * HEIGHT
                pygame.draw.line(screen, YELLOW, (cx - 2, y0), (cx - 2, y0 + HEIGHT), 2)
                pygame.draw.line(screen, YELLOW, (cx + 2, y0), (cx + 2, y0 + HEIGHT), 2)

        for off in (off1, off2):
            for dy in range(-1, 3):
                y0 = int(off) + dy * HEIGHT
                pygame.draw.line(screen, WHITE, (self.left_x,  y0), (self.left_x,  y0 + HEIGHT), 5)
                pygame.draw.line(screen, WHITE, (self.right_x, y0), (self.right_x, y0 + HEIGHT), 5)

        self._draw_arrows(screen, off1)

    def _draw_arrows(self, screen, off):
        spacing = 220
        aw, ah  = 14, 26
        col     = (75, 75, 85)
        lx = LANE_CENTERS[0]
        for raw_y in range(0, HEIGHT + spacing * 2, spacing):
            y = (raw_y + int(off)) % (HEIGHT + spacing) - spacing // 2
            pts = [(lx, y - ah // 2), (lx - aw // 2, y + ah // 2),
                   (lx, y + ah // 4), (lx + aw // 2, y + ah // 2)]
            pygame.draw.polygon(screen, col, pts)

        rx = LANE_CENTERS[1]
        ph = spacing // 2
        for raw_y in range(0, HEIGHT + spacing * 2, spacing):
            y = (raw_y + int(off) + ph) % (HEIGHT + spacing) - spacing // 2
            pts = [(rx, y + ah // 2), (rx - aw // 2, y - ah // 2),
                   (rx, y - ah // 4), (rx + aw // 2, y - ah // 2)]
            pygame.draw.polygon(screen, col, pts)


class Traffic:
    """
    Traffic manager.

    Same-direction cars (lane 0, going UP / going_down=False):
        Spawned above the screen and scroll down with the road.

    Oncoming cars (lane 1, going DOWN / going_down=True):
        Governed by a metered system with mandatory quiet windows so the
        ego car always gets clear stretches to practice overtaking.

    Quiet-window logic
    ------------------
    After each oncoming car leaves the screen a cooldown timer starts.
    During the cooldown the right lane is guaranteed empty.
    Cooldown duration is randomised between MIN_QUIET and MAX_QUIET frames
    so the driver sees a variety of gap lengths.
    """

    # Quiet window duration in frames @ 60 fps
    MIN_QUIET_FRAMES = 180   # 3 s minimum gap
    MAX_QUIET_FRAMES = 480   # 8 s maximum gap  → plenty of time to overtake

    def __init__(self):
        self._id_seq        = 1000
        self.cars           = []
        self._overtaken_ids = set()

        # Oncoming pacing state
        self._oncoming_cooldown = 0    # frames remaining in quiet window
        self._oncoming_active   = False  # True while a car is on screen

        # Post-overtake: no spawn immediately in front of ego
        self._post_overtake_cooldown = 0

        raw = load_image('traffic.png', (160, 60, 60), (50, 88))
        rw, rh = raw.get_size()
        self._car_w = CAR_WIDTH
        self._car_h = int(rh * (self._car_w / rw))
        self._img_up   = pygame.transform.scale(raw, (self._car_w, self._car_h))
        self._img_down = pygame.transform.flip(self._img_up, False, True)

        self._initial_populate()

    # ── helpers ────────────────────────────────────────────────────────
    def _next_id(self):
        i = self._id_seq; self._id_seq += 1; return i

    def _make_car(self, lane, y_float, speed, going_down):
        img = self._img_down if going_down else self._img_up
        x   = LANE_CENTERS[lane] - self._car_w // 2
        return {
            'id':         self._next_id(),
            'lane':       lane,
            'image':      img,
            'x':          float(x),
            'y':          float(y_float),
            'rect':       pygame.Rect(x, int(y_float), self._car_w, self._car_h),
            'speed':      speed,
            'going_down': going_down,
        }

    def _sync_rect(self, car):
        car['rect'].y = int(car['y'])

    # ── initial populate ───────────────────────────────────────────────
    def _initial_populate(self):
        # Same-dir cars staggered above screen
        y = -self._car_h - 80
        for _ in range(SAME_DIR_NUM_CARS):
            spd = random.uniform(SAME_DIR_SPEED_MIN, SAME_DIR_SPEED_MAX)
            self.cars.append(self._make_car(0, y, spd, False))
            y -= self._car_h + SAME_DIR_SPAWN_GAP + random.randint(0, 120)

        # Start with a long quiet window so the first overtake chance
        # arrives quickly and the player can see it clearly.
        self._oncoming_cooldown = self.MAX_QUIET_FRAMES
        self._oncoming_active   = False

    # ── main update ────────────────────────────────────────────────────
    def update(self, scroll, player_y):
        overtakes  = 0
        to_remove  = []

        for car in self.cars:
            if car['going_down']:
                car['y'] += car['speed'] + scroll
            else:
                car['y'] += scroll - car['speed']
                cid = car['id']
                if car['y'] > player_y + 20 and cid not in self._overtaken_ids:
                    self._overtaken_ids.add(cid)
                    overtakes += 1

            self._sync_rect(car)

            if car['rect'].top > HEIGHT + 300:
                to_remove.append(car)

        for car in to_remove:
            self.cars.remove(car)
            if car['id'] in self._overtaken_ids:
                self._overtaken_ids.discard(car['id'])
            if car['going_down']:
                self._oncoming_active   = False
                quiet = random.randint(self.MIN_QUIET_FRAMES, self.MAX_QUIET_FRAMES)
                self._oncoming_cooldown = quiet

        if overtakes > 0:
            self._post_overtake_cooldown = POST_OVERTAKE_SPAWN_COOLDOWN_FRAMES
        if self._post_overtake_cooldown > 0:
            self._post_overtake_cooldown -= 1

        self._enforce_same_dir_gaps()
        self._try_spawn_same_dir(player_y)
        self._tick_oncoming()

        return overtakes

    # ── oncoming metered spawner ───────────────────────────────────────
    def _tick_oncoming(self):
        """
        Metered oncoming spawner.
        Only one oncoming car is allowed on screen at a time.
        After it leaves, a randomised quiet window ensures the right lane
        is clear long enough for an overtake.
        """
        # Count active oncoming cars currently on screen
        onc_on_screen = [c for c in self.cars if c['going_down']]
        self._oncoming_active = len(onc_on_screen) > 0

        if self._oncoming_active:
            # A car is already on screen — do nothing
            return

        # Count-down the quiet window
        if self._oncoming_cooldown > 0:
            self._oncoming_cooldown -= 1
            return

        # Quiet window expired — spawn the next oncoming car
        spd = random.uniform(ONCOMING_SPEED_MIN, ONCOMING_SPEED_MAX)
        # Spawn just off the top of the screen
        spawn_y = -self._car_h - random.randint(40, 120)
        self.cars.append(self._make_car(1, spawn_y, spd, True))
        self._oncoming_active = True

    # ── same-direction gap enforcement ────────────────────────────────
    def _enforce_same_dir_gaps(self):
        same = sorted([c for c in self.cars if not c['going_down']], key=lambda c: c['y'])
        for i in range(1, len(same)):
            front = same[i - 1]
            rear  = same[i]
            gap = rear['y'] - (front['y'] + self._car_h)
            if gap < SAME_DIR_MIN_GAP:
                rear['y'] = front['y'] + self._car_h + SAME_DIR_MIN_GAP
                self._sync_rect(rear)

    def _try_spawn_same_dir(self, player_y):
        import config as cfg
        if random.random() > SAME_DIR_SPAWN_PROB:
            return
        same = [c for c in self.cars if not c['going_down']]
        if len(same) >= cfg.SAME_DIR_NUM_CARS:
            return
        if same:
            topmost_y = min(c['y'] for c in same)
            if topmost_y > -self._car_h - SAME_DIR_SPAWN_GAP + 100:
                return

        # Minimum distance ahead of ego: never spawn immediately in front (safety buffer + cooldown zone)
        buffer = POST_OVERTAKE_SPAWN_BUFFER if self._post_overtake_cooldown > 0 else MIN_SPAWN_DIST_AHEAD
        max_spawn_y = player_y - buffer - self._car_h
        y_candidate = -self._car_h - random.randint(30, 100)
        y = min(y_candidate, max_spawn_y)

        spd = random.uniform(SAME_DIR_SPEED_MIN, SAME_DIR_SPEED_MAX)
        self.cars.append(self._make_car(0, y, spd, False))

    # ── draw ──────────────────────────────────────────────────────────
    def draw(self, screen):
        for car in self.cars:
            screen.blit(car['image'], (int(car['x']), int(car['y'])))

    @property
    def car_height(self):
        return self._car_h

    # ── public info ───────────────────────────────────────────────────
    def get_oncoming_cooldown(self):
        """Return remaining quiet frames (0 when a car is spawning)."""
        return self._oncoming_cooldown