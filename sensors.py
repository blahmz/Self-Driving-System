"""
Sensor System — Multi-directional ray-casting
Adapted from reference project for two-lane bidirectional road.
"""

import pygame
import math
from config import SENSOR_CONFIG, YELLOW, CYAN, MAGENTA, RED_LASER, BLUE_LASER


def get_intersection(p1, p2, p3, p4):
    """Segment-segment intersection using linear algebra."""
    a1 = p2['y'] - p1['y']
    b1 = p1['x'] - p2['x']
    c1 = a1 * p1['x'] + b1 * p1['y']

    a2 = p4['y'] - p3['y']
    b2 = p3['x'] - p4['x']
    c2 = a2 * p3['x'] + b2 * p3['y']

    det = a1 * b2 - a2 * b1
    if det == 0:
        return None

    x = (b2 * c1 - b1 * c2) / det
    y = (a1 * c2 - a2 * c1) / det

    eps = 1e-9
    if (min(p1['x'], p2['x']) - eps <= x <= max(p1['x'], p2['x']) + eps and
        min(p1['y'], p2['y']) - eps <= y <= max(p1['y'], p2['y']) + eps and
        min(p3['x'], p4['x']) - eps <= x <= max(p3['x'], p4['x']) + eps and
        min(p3['y'], p4['y']) - eps <= y <= max(p3['y'], p4['y']) + eps):
        offset = math.hypot(x - p1['x'], y - p1['y'])
        return {'x': x, 'y': y, 'offset': offset}
    return None


class Sensor:
    """Full multi-directional sensor array identical in spirit to reference project."""

    def __init__(self, car):
        self.car = car
        self.rays    = []
        self.readings = []

    # ------------------------------------------------------------------ #
    def update(self, road_borders, traffic):
        self._cast_rays()
        self.readings = []
        for i, ray in enumerate(self.rays):
            reading = self._get_reading(ray, road_borders, traffic)
            if reading is None:
                self.readings.append({'offset': self._max_len(i)})
            else:
                self.readings.append(reading)

    # ------------------------------------------------------------------ #
    def _cast_rays(self):
        self.rays = []
        self._front_rays()
        self._diagonal_rays()
        self._rear_rays()

    def _front_rays(self):
        cfg    = SENSOR_CONFIG['front']
        count  = cfg['count']
        length = cfg['length']
        spread = math.radians(cfg['spread'])
        for i in range(count):
            t = i / (count - 1) if count > 1 else 0.5
            angle = _lerp(spread / 2, -spread / 2, t) + math.radians(self.car.angle)
            self._add_ray(angle, length, 'front')

    def _diagonal_rays(self):
        cfg    = SENSOR_CONFIG['diagonal']
        count  = cfg['count']
        length = cfg['length']
        spread = math.radians(cfg['spread'])
        offset = math.radians(cfg['offset'])
        base   = math.radians(self.car.angle)

        # Left diagonal
        for i in range(count):
            t = i / (count - 1) if count > 1 else 0.5
            angle = base + offset + _lerp(-spread / 2, spread / 2, t)
            self._add_ray(angle, length, 'side')
        # Right diagonal
        for i in range(count):
            t = i / (count - 1) if count > 1 else 0.5
            angle = base - offset + _lerp(spread / 2, -spread / 2, t)
            self._add_ray(angle, length, 'side')

    def _rear_rays(self):
        cfg_r  = SENSOR_CONFIG['rear']
        count  = cfg_r['count']
        length = cfg_r['length']
        spread = math.radians(cfg_r['spread'])
        base   = math.radians(self.car.angle) + math.pi

        for i in range(count):
            t = i / (count - 1) if count > 1 else 0.5
            angle = base + _lerp(-spread / 2, spread / 2, t)
            self._add_ray(angle, length, 'rear')

        cfg_rd = SENSOR_CONFIG['rear_diag']
        rd_ang = math.radians(cfg_rd['angle'])
        self._add_ray(base + rd_ang, cfg_rd['length'], 'rear')
        self._add_ray(base - rd_ang, cfg_rd['length'], 'rear')

    def _add_ray(self, angle, length, ray_type):
        from config import CAR_WIDTH
        cx = self.car.x + CAR_WIDTH // 2
        cy = self.car.y + self.car.car_height // 2

        if ray_type == 'rear':
            cy = self.car.y + self.car.car_height

        start = {'x': cx, 'y': cy}
        end   = {
            'x': cx - math.sin(angle) * length,
            'y': cy - math.cos(angle) * length,
        }
        self.rays.append([start, end])

    # ------------------------------------------------------------------ #
    def _front_count(self):
        return SENSOR_CONFIG['front']['count']

    def _diag_total(self):
        return SENSOR_CONFIG['diagonal']['count'] * 2

    def _rear_count(self):
        return SENSOR_CONFIG['rear']['count'] + SENSOR_CONFIG['rear_diag']['count']

    def _max_len(self, idx):
        fc = self._front_count()
        dt = self._diag_total()
        if idx < fc:
            return SENSOR_CONFIG['front']['length']
        elif idx < fc + dt:
            return SENSOR_CONFIG['diagonal']['length']
        else:
            return SENSOR_CONFIG['rear']['length']

    # ------------------------------------------------------------------ #
    def _get_reading(self, ray, road_borders, traffic):
        touches = []

        # Road borders
        for border in road_borders:
            if isinstance(border, list) and len(border) == 2:
                touch = get_intersection(ray[0], ray[1], border[0], border[1])
                if touch:
                    touches.append(touch)

        # All traffic cars (both lanes)
        for car in traffic.cars:
            poly = [
                {'x': car['rect'].left,  'y': car['rect'].top},
                {'x': car['rect'].right, 'y': car['rect'].top},
                {'x': car['rect'].right, 'y': car['rect'].bottom},
                {'x': car['rect'].left,  'y': car['rect'].bottom},
            ]
            for j in range(4):
                v = get_intersection(ray[0], ray[1], poly[j], poly[(j + 1) % 4])
                if v:
                    touches.append(v)

        if not touches:
            return None
        return min(touches, key=lambda t: t['offset'])

    # ------------------------------------------------------------------ #
    def draw(self, screen):
        fc = self._front_count()
        dt = self._diag_total()
        for i, ray in enumerate(self.rays):
            reading = self.readings[i] if i < len(self.readings) else None
            hit = reading is not None and 'x' in reading

            # Only draw when actively hitting an object — invisible otherwise
            if not hit:
                continue

            end_pt = reading

            if i < fc:
                dot_col = YELLOW
            elif i < fc + dt:
                dot_col = CYAN
            else:
                dot_col = MAGENTA

            # Red line from sensor origin to hit point
            pygame.draw.line(screen, RED_LASER,
                             (int(ray[0]['x']), int(ray[0]['y'])),
                             (int(end_pt['x']),  int(end_pt['y'])),
                             2)
            # Colored dot at hit point for easy identification
            pygame.draw.circle(screen, dot_col,
                               (int(end_pt['x']), int(end_pt['y'])), 5)

    # ------------------------------------------------------------------ #
    def get_normalized_readings(self):
        norm = []
        for i, r in enumerate(self.readings):
            max_l = self._max_len(i)
            norm.append(r['offset'] / max_l if max_l > 0 else 1.0)
        return norm


def _lerp(a, b, t):
    return a + (b - a) * t
