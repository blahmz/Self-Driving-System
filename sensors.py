"""
Advanced Sensor System
Multi-directional ray-casting sensors with collision detection
"""

import pygame
import math
from config import SENSOR_CONFIG, YELLOW, CYAN, MAGENTA, RED_LASER


def get_intersection(p1, p2, p3, p4):
    """Calculate intersection point of two line segments"""
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
    
    # Check if intersection is within both segments
    if (min(p1['x'], p2['x']) <= x <= max(p1['x'], p2['x']) and
        min(p1['y'], p2['y']) <= y <= max(p1['y'], p2['y']) and
        min(p3['x'], p4['x']) <= x <= max(p3['x'], p4['x']) and
        min(p3['y'], p4['y']) <= y <= max(p3['y'], p4['y'])):
        
        offset = ((x - p1['x']) ** 2 + (y - p1['y']) ** 2) ** 0.5
        return {'x': x, 'y': y, 'offset': offset}
    
    return None


class Sensor:
    """Advanced multi-directional sensor system"""
    
    def __init__(self, car):
        self.car = car
        
        # Load sensor configuration
        self.front_config = SENSOR_CONFIG['front']
        self.diag_config = SENSOR_CONFIG['diagonal']
        self.rear_config = SENSOR_CONFIG['rear']
        self.rear_diag_config = SENSOR_CONFIG['rear_diagonal']
        
        self.rays = []
        self.readings = []
        
    def update(self, road_borders, traffic):
        """Update all sensor readings"""
        self._cast_rays()
        self.readings = []
        
        for i, ray in enumerate(self.rays):
            reading = self._get_reading(ray, road_borders, traffic)
            
            if reading is None:
                # Use max length based on ray type
                max_len = self._get_max_length_for_ray(i)
                self.readings.append({'offset': max_len})
            else:
                self.readings.append(reading)
    
    def _get_max_length_for_ray(self, ray_index):
        """Get maximum length for specific ray"""
        front_count = self.front_config['count']
        diag_total = self.diag_config['count'] * 2
        rear_count = self.rear_config['count']
        
        if ray_index < front_count:
            return self.front_config['length']
        elif ray_index < front_count + diag_total:
            return self.diag_config['length']
        else:
            return self.rear_config['length']
    
    def _cast_rays(self):
        """Cast all sensor rays"""
        self.rays = []
        
        # Front sensors
        self._cast_front_rays()
        
        # Diagonal sensors
        self._cast_diagonal_rays()
        
        # Rear sensors
        self._cast_rear_rays()
    
    def _cast_front_rays(self):
        """Cast forward-facing rays"""
        count = self.front_config['count']
        length = self.front_config['length']
        spread = math.radians(self.front_config['spread'])
        
        for i in range(count):
            ray_angle = self._lerp(
                spread / 2,
                -spread / 2,
                i / (count - 1) if count > 1 else 0.5
            ) + math.radians(self.car.angle)
            
            self._add_ray(ray_angle, length, "front")
    
    def _cast_diagonal_rays(self):
        """Cast side diagonal rays"""
        count = self.diag_config['count']
        length = self.diag_config['length']
        spread = math.radians(self.diag_config['spread'])
        offset = math.radians(self.diag_config['offset'])
        
        # Left diagonal
        for i in range(count):
            base_angle = math.radians(self.car.angle) + offset
            ray_angle = base_angle + self._lerp(
                -spread / 2,
                spread / 2,
                i / (count - 1) if count > 1 else 0.5
            )
            self._add_ray(ray_angle, length, "side")
        
        # Right diagonal
        for i in range(count):
            base_angle = math.radians(self.car.angle) - offset
            ray_angle = base_angle + self._lerp(
                spread / 2,
                -spread / 2,
                i / (count - 1) if count > 1 else 0.5
            )
            self._add_ray(ray_angle, length, "side")
    
    def _cast_rear_rays(self):
        """Cast rear-facing rays"""
        count = self.rear_config['count']
        length = self.rear_config['length']
        spread = math.radians(self.rear_config['spread'])
        
        # Straight back
        for i in range(count):
            base_angle = math.radians(self.car.angle) + math.pi
            ray_angle = base_angle + self._lerp(
                -spread / 2,
                spread / 2,
                i / (count - 1) if count > 1 else 0.5
            )
            self._add_ray(ray_angle, length, "rear")
        
        # Rear diagonal corners
        angle_left = math.radians(self.car.angle) + math.pi + math.radians(
            self.rear_diag_config['angle']
        )
        self._add_ray(angle_left, self.rear_diag_config['length'], "rear")
        
        angle_right = math.radians(self.car.angle) + math.pi - math.radians(
            self.rear_diag_config['angle']
        )
        self._add_ray(angle_right, self.rear_diag_config['length'], "rear")
    
    def _add_ray(self, angle, length, ray_type):
        """Add a ray to the sensor array"""
        from config import CAR_WIDTH
        
        # Adjust start position based on ray type
        if ray_type == "rear":
            start = {
                "x": self.car.x + CAR_WIDTH // 2,
                "y": self.car.y + self.car.car_height
            }
        else:
            start = {
                "x": self.car.x + CAR_WIDTH // 2,
                "y": self.car.y + self.car.car_height // 2
            }
        
        end = {
            "x": start["x"] - math.sin(angle) * length,
            "y": start["y"] - math.cos(angle) * length
        }
        
        self.rays.append([start, end])
    
    def _lerp(self, start, end, t):
        """Linear interpolation"""
        return start + (end - start) * t
    
    def _get_reading(self, ray, road_borders, traffic):
        """Get sensor reading for a single ray"""
        touches = []
        
        # Check road borders
        for border in road_borders:
            if isinstance(border, list) and len(border) == 2:
                touch = get_intersection(ray[0], ray[1], border[0], border[1])
                if touch:
                    touches.append(touch)
        
        # Check traffic vehicles
        for car in traffic.cars:
            poly = [
                {"x": car['rect'].left, "y": car['rect'].top},
                {"x": car['rect'].right, "y": car['rect'].top},
                {"x": car['rect'].right, "y": car['rect'].bottom},
                {"x": car['rect'].left, "y": car['rect'].bottom}
            ]
            
            for j in range(len(poly)):
                value = get_intersection(
                    ray[0], ray[1],
                    poly[j], poly[(j + 1) % len(poly)]
                )
                if value:
                    touches.append(value)
        
        if not touches:
            return None
        
        return min(touches, key=lambda t: t["offset"])
    
    def draw(self, screen):
        """Visualize all sensors"""
        front_count = self.front_config['count']
        diag_total = self.diag_config['count'] * 2
        
        for i, ray in enumerate(self.rays):
            end = self.readings[i] if self.readings[i] and 'x' in self.readings[i] else ray[1]
            
            # Determine ray type and color
            if i < front_count:
                base_color = YELLOW
            elif i < front_count + diag_total:
                base_color = CYAN
            else:
                base_color = MAGENTA
            
            # Check if obstacle detected
            dist_to_end = ((end['x'] - ray[1]['x'])**2 + (end['y'] - ray[1]['y'])**2)**0.5
            
            if dist_to_end > 1.0:
                color = RED_LASER
                thickness = 2
            else:
                color = base_color
                thickness = 1
            
            # Draw ray
            pygame.draw.line(
                screen,
                color,
                (ray[0]['x'], ray[0]['y']),
                (end['x'], end['y']),
                thickness
            )
            
            # Draw hit point
            if color == RED_LASER:
                pygame.draw.circle(screen, RED_LASER, (int(end['x']), int(end['y'])), 4)
    
    def get_normalized_readings(self):
        """Get sensor readings normalized to [0, 1]"""
        normalized = []
        
        for i, reading in enumerate(self.readings):
            max_len = self._get_max_length_for_ray(i)
            normalized_value = reading['offset'] / max_len
            normalized.append(normalized_value)
        
        return normalized