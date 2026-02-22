"""
Vehicle State Management
Tracks ego state history and surrounding vehicle positions
"""

import numpy as np
from collections import deque
from config import SEQUENCE_LENGTH


class VehicleState:
    """Tracks and maintains ego vehicle state over time"""

    def __init__(self):
        self.history = deque(maxlen=SEQUENCE_LENGTH)
        self.x = 0.0
        self.y = 0.0
        self.angle = 0.0
        self.speed = 0.0
        self.lane  = 0
        self.last_action = "forward"

        # Velocity tracking
        self._prev_x = 0.0
        self._prev_y = 0.0
        self._prev_speed = 0.0

    def update(self, x, y, angle, speed, lane, sensor_readings, action):
        """Update state with new frame data"""
        self.x     = x
        self.y     = y
        self.angle = angle
        self.speed = speed
        self.lane  = lane
        self.last_action = action

        # Compute velocity
        vel_x = x - self._prev_x
        vel_y = y - self._prev_y
        accel = speed - self._prev_speed

        self._prev_x = x
        self._prev_y = y
        self._prev_speed = speed

        # Store snapshot
        state_snapshot = {
            'position':     {'x': x, 'y': y},
            'velocity':     {'x': vel_x, 'y': vel_y},
            'acceleration': {'x': 0.0,   'y': accel},
            'angle':        angle,
            'speed':        speed,
            'lane':         lane,
            'sensor_readings': list(sensor_readings),
            'action':       action,
        }
        self.history.append(state_snapshot)

    def get_state_vector(self):
        """Return latest state dict for FSM / safety analysis"""
        if self.history:
            return self.history[-1]
        return {
            'position':     {'x': self.x, 'y': self.y},
            'velocity':     {'x': 0.0, 'y': 0.0},
            'acceleration': {'x': 0.0, 'y': 0.0},
            'angle':        self.angle,
            'speed':        self.speed,
            'lane':         self.lane,
            'sensor_readings': [],
            'action':       self.last_action,
            'target_lane':  self.lane,
        }

    def get_sequence_data(self):
        """Return (SEQUENCE_LENGTH, num_sensors) array for LSTM"""
        if not self.history:
            return None
        rows = []
        for snap in self.history:
            rows.append(snap['sensor_readings'])
        arr = np.array(rows, dtype=np.float32)
        return arr


# ---------------------------------------------------------------------------
class SurroundingVehicles:
    """Tracks nearby traffic cars for FSM / safety analysis"""

    def __init__(self):
        self._vehicles = []     # list of normalised vehicle dicts

    def update(self, traffic_cars, ego_pos):
        """Rebuild vehicle list from raw traffic data each frame.
        traffic_cars: list of dicts with keys 'id','lane','rect','speed'
        ego_pos: {'x': float, 'y': float}
        """
        self._vehicles = []
        for car in traffic_cars:
            cx = car['rect'].centerx
            cy = car['rect'].centery
            spd = car.get('speed', 0)
            going_down = car.get('going_down', False)

            # velocity: same-dir cars move UP (negative vy in screen coords),
            # oncoming cars move DOWN (positive vy)
            vy = spd if going_down else -spd

            self._vehicles.append({
                'id':       car['id'],
                'lane':     car['lane'],
                'position': {'x': float(cx), 'y': float(cy)},
                'velocity': vy,
                'rect':     car['rect'],
                'going_down': going_down,
            })

    # ── queries ────────────────────────────────────────────────────────
    def get_vehicle_count(self):
        return len(self._vehicles)

    def get_vehicles_by_lane(self, lane):
        return [v for v in self._vehicles if v['lane'] == lane]

    def get_closest_vehicle_ahead(self, lane, ego_y):
        """Return closest vehicle ABOVE ego (lower screen y) in given lane."""
        candidates = [v for v in self._vehicles
                      if v['lane'] == lane and v['position']['y'] < ego_y]
        if not candidates:
            return None
        return max(candidates, key=lambda v: v['position']['y'])

    def get_closest_vehicle_behind(self, lane, ego_y):
        """Return closest vehicle BELOW ego (higher screen y) in given lane."""
        candidates = [v for v in self._vehicles
                      if v['lane'] == lane and v['position']['y'] > ego_y]
        if not candidates:
            return None
        return min(candidates, key=lambda v: v['position']['y'])

    def get_gap_to_car_ahead(self, lane, ego_y, car_height):
        """Pixel gap between ego front and nearest car ahead in lane."""
        car = self.get_closest_vehicle_ahead(lane, ego_y)
        if car is None:
            return float('inf')
        return ego_y - (car['rect'].bottom)

    def is_lane_clear_for_overtake(self, ego_y, look_ahead, look_behind):
        """
        Check right lane (lane 1) for oncoming cars.
        Returns True if no oncoming car is within look_ahead px ahead
        or look_behind px behind ego.
        """
        for v in self._vehicles:
            if v['lane'] != 1:
                continue
            vy = v['position']['y']
            # Oncoming car is above ego (coming toward us)
            if vy < ego_y + look_ahead and vy > ego_y - look_behind:
                return False
        return True
