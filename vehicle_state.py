"""
Vehicle State Management
Tracks complete vehicle state including position, velocity, and temporal history
"""

import numpy as np
from collections import deque
from config import SEQUENCE_LENGTH, TIME_PER_FRAME


class VehicleState:
    """Maintains comprehensive vehicle state with temporal history"""
    
    def __init__(self):
        # Current state
        self.position = {'x': 0, 'y': 0}
        self.velocity = {'x': 0, 'y': 0}
        self.acceleration = {'x': 0, 'y': 0}
        self.angle = 0
        self.speed = 0
        self.lane = 1
        
        # Temporal history (for LSTM)
        self.position_history = deque(maxlen=SEQUENCE_LENGTH)
        self.velocity_history = deque(maxlen=SEQUENCE_LENGTH)
        self.sensor_history = deque(maxlen=SEQUENCE_LENGTH)
        self.action_history = deque(maxlen=SEQUENCE_LENGTH)
        
        # Previous state for derivatives
        self.prev_position = {'x': 0, 'y': 0}
        self.prev_velocity = {'x': 0, 'y': 0}
        self.prev_speed = 0
        
        # Steering state
        self.steering_angle = 0
        self.target_lane = 1
        
    def update(self, x, y, angle, speed, lane, sensor_readings, action=None):
        """Update all state variables"""
        # Store previous values
        self.prev_position = self.position.copy()
        self.prev_velocity = self.velocity.copy()
        self.prev_speed = self.speed
        
        # Update current state
        self.position = {'x': x, 'y': y}
        self.angle = angle
        self.speed = speed
        self.lane = lane
        
        # Calculate velocity (pixels per second)
        dt = TIME_PER_FRAME
        self.velocity['x'] = (x - self.prev_position['x']) / dt if dt > 0 else 0
        self.velocity['y'] = (y - self.prev_position['y']) / dt if dt > 0 else 0
        
        # Calculate acceleration
        self.acceleration['x'] = (self.velocity['x'] - self.prev_velocity['x']) / dt if dt > 0 else 0
        self.acceleration['y'] = (self.velocity['y'] - self.prev_velocity['y']) / dt if dt > 0 else 0
        
        # Update history
        self.position_history.append([x, y])
        self.velocity_history.append([self.velocity['x'], self.velocity['y']])
        self.sensor_history.append(sensor_readings.copy())
        if action:
            self.action_history.append(action)
    
    def get_sequence_data(self):
        """Get temporal sequence for LSTM input"""
        if len(self.sensor_history) < SEQUENCE_LENGTH:
            # Pad with zeros if not enough history
            padding_needed = SEQUENCE_LENGTH - len(self.sensor_history)
            padded_sensors = [np.zeros_like(self.sensor_history[0])] * padding_needed
            padded_sensors.extend(list(self.sensor_history))
            return np.array(padded_sensors)
        else:
            return np.array(list(self.sensor_history))
    
    def get_state_vector(self):
        """Get current state as vector for analysis"""
        return {
            'position': self.position,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'speed': self.speed,
            'angle': self.angle,
            'lane': self.lane,
            'steering_angle': self.steering_angle
        }
    
    def reset_history(self):
        """Clear temporal history"""
        self.position_history.clear()
        self.velocity_history.clear()
        self.sensor_history.clear()
        self.action_history.clear()


class SurroundingVehicles:
    """Manages state of surrounding vehicles"""
    
    def __init__(self):
        self.vehicles = []
        self.history = deque(maxlen=SEQUENCE_LENGTH)
        
    def update(self, traffic_cars, ego_position):
        """Update surrounding vehicle states"""
        self.vehicles = []
        
        for car in traffic_cars:
            vehicle_info = {
                'id': car['id'],
                'position': {'x': car['rect'].centerx, 'y': car['rect'].centery},
                'velocity': car.get('speed', 0),
                'lane': car['lane'],
                'width': car['rect'].width,
                'height': car['rect'].height,
                'relative_x': car['rect'].centerx - ego_position['x'],
                'relative_y': car['rect'].centery - ego_position['y']
            }
            self.vehicles.append(vehicle_info)
        
        # Store in history
        self.history.append(self.vehicles.copy())
    
    def get_vehicle_count(self):
        """Get total number of surrounding vehicles"""
        return len(self.vehicles)
    
    def get_vehicles_by_lane(self, lane):
        """Get vehicles in specific lane"""
        return [v for v in self.vehicles if v['lane'] == lane]
    
    def get_closest_vehicle_ahead(self, ego_lane, ego_y):
        """Get closest vehicle ahead in same lane"""
        lane_vehicles = self.get_vehicles_by_lane(ego_lane)
        ahead = [v for v in lane_vehicles if v['position']['y'] < ego_y]
        
        if not ahead:
            return None
        
        return min(ahead, key=lambda v: ego_y - v['position']['y'])
    
    def get_closest_vehicle_behind(self, ego_lane, ego_y):
        """Get closest vehicle behind in same lane"""
        lane_vehicles = self.get_vehicles_by_lane(ego_lane)
        behind = [v for v in lane_vehicles if v['position']['y'] > ego_y]
        
        if not behind:
            return None
        
        return min(behind, key=lambda v: v['position']['y'] - ego_y)
    
    def get_temporal_sequence(self):
        """Get temporal sequence of vehicle states"""
        if len(self.history) < SEQUENCE_LENGTH:
            padding_needed = SEQUENCE_LENGTH - len(self.history)
            padded = [[]] * padding_needed
            padded.extend(list(self.history))
            return padded
        else:
            return list(self.history)