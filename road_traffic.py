"""
Road and Traffic Management
Handles road rendering and dynamic traffic generation
"""

import pygame
import random
import os
from config import (WIDTH, HEIGHT, ROAD_WIDTH, LANE_COUNT, LANE_WIDTH, WHITE)


def load_image(name, fallback_color=(100, 100, 100)):
    """Safe image loader"""
    if os.path.exists(name):
        return pygame.image.load(name)
    elif os.path.exists(name.replace(".jpg", ".png")):
        return pygame.image.load(name.replace(".jpg", ".png"))
    else:
        surface = pygame.Surface((50, 50))
        surface.fill(fallback_color)
        return surface


class Road:
    """Road rendering and management"""
    
    def __init__(self, x, width, lane_count, grass_texture):
        self.x = x
        self.width = width
        self.lane_count = lane_count
        
        # Boundaries
        self.left = x - width / 2
        self.right = x + width / 2
        self.top = -1000000
        self.bottom = 1000000
        
        # Borders for collision detection
        self.borders = [
            [{'x': self.left, 'y': self.top}, {'x': self.left, 'y': self.bottom}],
            [{'x': self.right, 'y': self.top}, {'x': self.right, 'y': self.bottom}]
        ]
        
        # Textures
        self.grass_texture = grass_texture
        
        # Load road texture
        road_texture_img = load_image("road_texture.jpg")
        self.road_texture = pygame.transform.scale(road_texture_img, (width, HEIGHT))
        
        # Lane positions (for traffic spawning)
        self.lane_positions = [
            self.left + (self.width / self.lane_count) * (i + 0.5) 
            for i in range(self.lane_count)
        ]
    
    def draw(self, screen, road_offset_1, road_offset_2):
        """Draw road with scrolling effect"""
        # Draw grass
        if self.grass_texture:
            for i in range(-2, 3):
                # Left grass
                screen.blit(self.grass_texture, (0, int(road_offset_1) + i * HEIGHT))
                screen.blit(self.grass_texture, (0, int(road_offset_2) + i * HEIGHT))
                
                # Right grass
                grass_x = WIDTH - (WIDTH - int(self.right))
                screen.blit(self.grass_texture, (grass_x, int(road_offset_1) + i * HEIGHT))
                screen.blit(self.grass_texture, (grass_x, int(road_offset_2) + i * HEIGHT))
        
        # Draw road texture
        for i in range(-2, 3):
            screen.blit(self.road_texture, (int(self.left), int(road_offset_1) + i * HEIGHT))
            screen.blit(self.road_texture, (int(self.left), int(road_offset_2) + i * HEIGHT))
        
        # Draw lane markings
        dash_gap = 40
        dash_length = 20
        
        for i in range(1, self.lane_count):
            lane_center_x = self.left + (self.width / self.lane_count) * i
            
            # Draw dashes for both scrolling textures
            for offset in [road_offset_1, road_offset_2]:
                for j in range(int(offset), int(offset + HEIGHT), dash_gap + dash_length):
                    if j < self.bottom:
                        pygame.draw.line(
                            screen, WHITE,
                            (lane_center_x, j),
                            (lane_center_x, j + dash_length),
                            4
                        )
        
        # Draw borders
        for border in self.borders:
            pygame.draw.line(
                screen, WHITE,
                (border[0]['x'], border[0]['y']),
                (border[1]['x'], border[1]['y']),
                6
            )


class Traffic:
    """Traffic vehicle management with density control"""
    
    def __init__(self, num_cars, lane_width, num_lanes, road_width, road, density_manager):
        self.num_cars = num_cars
        self.lane_width = lane_width
        self.num_lanes = num_lanes
        self.road_width = road_width
        self.road = road
        self.density_manager = density_manager
        
        self.cars = []
        self.overtaken_cars = set()
        
        # Load traffic car image
        self.car_image = load_image('traffic.png', (255, 0, 0))
        
        # Initialize traffic
        self.create_cars()
    
    def create_cars(self):
        """Create initial traffic vehicles"""
        for i in range(self.num_cars):
            self._spawn_new_car(initial_spawn=True, vertical_offset=i * 250)
    
    def _spawn_new_car(self, initial_spawn=False, vertical_offset=0):
        """Spawn a new traffic vehicle"""
        # Get density parameters
        density_params = self.density_manager.get_spawn_parameters()
        max_cars = density_params['num_cars']
        min_gap = density_params['min_gap']
        
        # Check if we've reached max cars
        if len(self.cars) >= max_cars:
            return
        
        # Find occupied lanes
        occupied_lanes = {car['lane'] for car in self.cars}
        available_lanes = [i for i in range(self.num_lanes) if i not in occupied_lanes]
        
        if len(available_lanes) <= 1:
            return
        
        # Choose random available lane
        lane_index = random.choice(available_lanes)
        
        # Scale car image
        car_width, car_height = self.car_image.get_size()
        scale_factor = random.uniform(0.55, 0.65)
        new_width = int(self.lane_width * scale_factor)
        new_height = int(car_height * (new_width / car_width))
        car_image = pygame.transform.scale(self.car_image, (new_width, new_height))
        
        # Position
        car_x = self.road.lane_positions[lane_index] - new_width // 2
        
        if initial_spawn:
            car_y = -new_height - vertical_offset
        else:
            car_y = -new_height - random.randint(100, 300)
        
        # Create car data
        car_id = random.randint(1000, 99999)
        speed = random.uniform(2.0, 5.0)
        
        self.cars.append({
            'id': car_id,
            'image': car_image,
            'rect': pygame.Rect(car_x, car_y, new_width, new_height),
            'speed': speed,
            'lane': lane_index
        })
    
    def update(self, road_scroll_speed, player_y):
        """Update all traffic vehicles"""
        density_params = self.density_manager.get_spawn_parameters()
        spawn_prob = density_params['spawn_prob']
        
        cars_to_remove = []
        overtakes_this_frame = 0
        
        for car in self.cars:
            # Random speed variation
            if random.random() < 0.02:
                speed_change = random.uniform(-0.1, 0.1)
                car['speed'] += speed_change
                car['speed'] = max(1.5, min(car['speed'], 5.5))
            
            # Update position
            car['rect'].y -= car['speed'] - road_scroll_speed
            
            # Check overtake
            if car['rect'].y > player_y and car['id'] not in self.overtaken_cars:
                self.overtaken_cars.add(car['id'])
                overtakes_this_frame += 1
            
            # Remove off-screen cars
            if car['rect'].y > HEIGHT + 100:
                cars_to_remove.append(car)
        
        # Remove cars
        for car in cars_to_remove:
            if car in self.cars:
                self.cars.remove(car)
                if car['id'] in self.overtaken_cars:
                    self.overtaken_cars.remove(car['id'])
        
        # Spawn new cars based on density
        if random.random() < spawn_prob:
            self._spawn_new_car()
        
        return overtakes_this_frame
    
    def draw(self, screen):
        """Draw all traffic vehicles"""
        for car in self.cars:
            screen.blit(car['image'], car['rect'].topleft)
    
    def update_density(self, new_num_cars):
        """Update target number of cars"""
        self.num_cars = new_num_cars