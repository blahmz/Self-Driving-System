"""
Main Vehicle Controller
Integrates sensors, state management, and control logic
"""

import pygame
import math
import time
import numpy as np
import os

from config import (CAR_WIDTH, MAX_ROTATION_ANGLE, ROTATION_SPEED, 
                    MAX_SPEED, ACCELERATION, DECELERATION, WIDTH, ROAD_WIDTH,
                    LANE_COUNT, LANE_WIDTH, HEIGHT, DECISION_FREQUENCY,
                    TTC_WARNING, TTC_CRITICAL, COLLISION_PROBABILITY_THRESHOLD)
from sensors import Sensor
from vehicle_state import VehicleState


def load_image(name, fallback_color=(100, 100, 100)):
    """Safe image loader with fallback"""
    if os.path.exists(name):
        return pygame.image.load(name)
    elif os.path.exists(name.replace(".jpg", ".png")):
        return pygame.image.load(name.replace(".jpg", ".png"))
    else:
        surface = pygame.Surface((50, 50))
        surface.fill(fallback_color)
        return surface


class Car:
    """Main autonomous vehicle with integrated control"""
    
    def __init__(self, road, traffic):
        # Load and scale image
        self.image = load_image("car.png", (255, 255, 0))
        self.original_width, self.original_height = self.image.get_size()
        aspect_ratio = self.original_height / self.original_width
        self.car_height = int(CAR_WIDTH * aspect_ratio)
        self.image = pygame.transform.scale(self.image, (CAR_WIDTH, self.car_height))
        
        # Position
        self.x = (WIDTH // 2) - (CAR_WIDTH // 2)
        self.y = HEIGHT - self.car_height - 100
        
        # Physics
        self.speed = 0
        self.angle = 0
        
        # Controls
        self.controls = {
            'forward': False,
            'reverse': False,
            'left': False,
            'right': False
        }
        
        # References
        self.road = road
        self.traffic = traffic
        
        # Sensor system
        self.sensor = Sensor(self)
        
        # State management
        self.vehicle_state = VehicleState()
        
        # Teacher mode state
        self.target_lane = 1
        self.lane_change_cooldown = 0
        
        # Decision timing
        self.decision_counter = 0

        # Persist AI command between decision frames (prevents "stutter/slowdown")
        self.last_ai_command = "forward"

        # Lane-change commitment (prevents oscillation / corner clipping)
        self.lane_change_active = False
        self.lane_change_dir = None  # "left" | "right" | None
        self.lane_change_target_lane = None
        self.lane_change_frames = 0
        self.lane_change_max_frames = 45
        
    def move(self, is_ai=False, model=None, encoder=None, scaler=None, 
             is_teacher=False, fsm_controller=None, safety_analyzer=None):
        """Main movement logic with multiple control modes"""
        
        # Reset controls
        self.controls = {
            'forward': False,
            'reverse': False,
            'left': False,
            'right': False
        }
        
        # Update decision counter
        self.decision_counter += 1
        
        # Get current lane
        current_lane = self._get_current_lane()
        
        # Update vehicle state
        sensor_readings = [r['offset'] for r in self.sensor.readings]
        action = self._get_current_action()
        self.vehicle_state.update(
            self.x, self.y, self.angle, self.speed, 
            current_lane, sensor_readings, action
        )
        
        # === CONTROL MODE SELECTION ===
        
        if is_teacher:
            # Teacher mode (sequential lane changing)
            self._teacher_mode_control()
            
        elif is_ai and model is not None and encoder is not None and scaler is not None:
            # AI mode (safety-aware decision frequency + persist last command)
            decision_period = DECISION_FREQUENCY
            if safety_analyzer is not None:
                forward_ttc = safety_analyzer.action_ttc.get('forward', float('inf'))
                forward_prob = safety_analyzer.action_collision_prob.get('forward', 0.0)
                # When close to the lead car, decide every frame to avoid "last-second" crashes.
                if forward_ttc < TTC_WARNING or forward_prob > 0.25:
                    decision_period = 1

            if self.decision_counter % decision_period == 0:
                self.last_ai_command = self._ai_mode_control(
                    model, encoder, scaler, fsm_controller, safety_analyzer
                )
            # Commit lane changes until we reach lane center.
            self.last_ai_command = self._apply_lane_change_commitment(
                self.last_ai_command, safety_analyzer
            )

            # Always apply *some* command every frame (otherwise we decelerate to 0)
            self._apply_action(self.last_ai_command)
            
        else:
            # Manual mode
            self._manual_mode_control()
        
        # Apply physics and movement
        success = self._apply_physics()
        
        if not success:
            return False
        
        # Update sensors
        self.sensor.update(self.road.borders, self.traffic)
        
        return True

    def _lane_center_x(self, lane_index: int) -> float:
        road_left_edge = (WIDTH - ROAD_WIDTH) // 2
        lane_width = ROAD_WIDTH / LANE_COUNT
        return road_left_edge + (lane_index * lane_width) + (lane_width / 2) - (CAR_WIDTH / 2)

    def _is_centered_in_lane(self, lane_index: int, tolerance_px: float = 18.0) -> bool:
        return abs(self.x - self._lane_center_x(lane_index)) <= tolerance_px

    def _apply_lane_change_commitment(self, desired_command: str, safety_analyzer=None) -> str:
        """
        If a lane change starts, keep steering that direction until centered
        in the target lane. Also apply an extra brake safety check during the maneuver.
        """
        current_lane = self._get_current_lane()

        # Start commitment if asked to change lanes and it's a legal lane.
        if not self.lane_change_active and desired_command in ("left", "right"):
            target_lane = current_lane + (-1 if desired_command == "left" else 1)
            if 0 <= target_lane < LANE_COUNT:
                self.lane_change_active = True
                self.lane_change_dir = desired_command
                self.lane_change_target_lane = target_lane
                self.lane_change_frames = 0

        # If committing, override command until completion/timeout.
        if self.lane_change_active:
            self.lane_change_frames += 1

            # Abort if somehow target invalid
            if self.lane_change_target_lane is None or not (0 <= self.lane_change_target_lane < LANE_COUNT):
                self.lane_change_active = False
                self.lane_change_dir = None
                self.lane_change_target_lane = None
                return "slow_down"

            # Extra safety during lane change: if forward is getting dangerous, brake.
            if safety_analyzer is not None:
                f_ttc = safety_analyzer.action_ttc.get("forward", float("inf"))
                f_prob = safety_analyzer.action_collision_prob.get("forward", 0.0)
                if f_ttc < TTC_WARNING or f_prob > 0.45:
                    return "slow_down"

            # Finish if centered.
            if self._is_centered_in_lane(self.lane_change_target_lane):
                self.lane_change_active = False
                self.lane_change_dir = None
                self.lane_change_target_lane = None
                return "forward"

            # Timeout: give up and slow down to recover safely.
            if self.lane_change_frames > self.lane_change_max_frames:
                self.lane_change_active = False
                self.lane_change_dir = None
                self.lane_change_target_lane = None
                return "slow_down"

            # Continue steering in the committed direction, but keep speed modest.
            # (reduces rear-corner clipping when passing close)
            if self.speed > 3.0:
                return "slow_down"
            return self.lane_change_dir

        return desired_command
    
    def _teacher_mode_control(self):
        """Teacher mode - Sequential lane change logic"""
        # Cooldown management
        if self.lane_change_cooldown > 0:
            self.lane_change_cooldown -= 1
        else:
            self._update_target_lane()
        
        # Execute lane centering
        action = self._execute_lane_centering()
        self._apply_action(action)
    
    def _ai_mode_control(self, model, encoder, scaler, fsm_controller, safety_analyzer):
        """AI mode with model + optional FSM/safety. Returns chosen command string."""
        def _decode_action(enc, idx: int) -> str:
            """Decode a predicted class index into an action string."""
            if enc is None:
                return "forward"
            if hasattr(enc, "inverse_transform"):
                return enc.inverse_transform([idx])[0]
            if hasattr(enc, "classes_"):
                return enc.classes_[idx]
            # External labels from `label_classes.npy` are typically a numpy array of strings
            try:
                return enc[idx]
            except Exception:
                return "forward"

        # If the loaded model is the external `autonomous_car` model, it expects
        # a single-timestep vector of 11 sensor distances: shape (1, 1, 11).
        use_external_format = False
        try:
            input_shape = getattr(model, "input_shape", None)
            if isinstance(input_shape, (list, tuple)) and len(input_shape) == 3:
                # (batch, timesteps, features)
                use_external_format = (input_shape[-1] == 11)
        except Exception:
            use_external_format = False

        if use_external_format:
            # Ensure sensors are up-to-date before reading.
            if not self.sensor.readings:
                self.sensor.update(self.road.borders, self.traffic)

            # Map this project's sensor system -> 11-dim distances in [0, 300],
            # matching the `autonomous_car` model's expected ray length (300px).
            # Use normalized readings per-ray, then scale to 300.
            normalized = []
            try:
                normalized = self.sensor.get_normalized_readings()
            except Exception:
                normalized = []

            # Simple rule-based overtaking trigger (sensor-driven).
            # This prevents "only slowing down" when the learned model is conservative.
            # Sensor order: front(5) then diagonal(left 3 + right 3) => indices:
            # - front_mid = 2
            # - left_diag = 5..7
            # - right_diag = 8..10
            front_mid = normalized[2] if len(normalized) > 2 else 1.0
            left_clear = min(normalized[5:8]) if len(normalized) >= 8 else 0.0
            right_clear = min(normalized[8:11]) if len(normalized) >= 11 else 0.0

            feats = [(float(v) if v is not None else 1.0) * 300.0 for v in normalized[:11]]
            if len(feats) < 11:
                feats = feats + [300.0] * (11 - len(feats))

            x = np.array(feats, dtype=np.float32).reshape(1, -1)
            x_scaled = scaler.transform(x) if scaler is not None else x
            x_scaled = x_scaled.reshape(1, 1, -1)

            pred = model.predict(x_scaled, verbose=0)
            predicted_index = int(np.argmax(pred))
            command = _decode_action(encoder, predicted_index)

            # If obstacle ahead is close, actively prefer a safe lane change.
            # Thresholds are in normalized [0..1] (smaller = closer obstacle).
            #
            # IMPORTANT: prevent "rear-corner clipping" crashes by requiring a minimum
            # longitudinal gap before initiating a lane change. If we're too close,
            # brake first; once we have a bit more spacing, then start the lane change.
            if front_mid < 0.55:
                lane = self._get_current_lane()
                prefer_left = (lane > 0) and (left_clear > right_clear) and (left_clear > 0.65)
                prefer_right = (lane < LANE_COUNT - 1) and (right_clear > 0.65)

                # Too close to the lead car: do NOT start a lateral move.
                # This is where we tend to hit the rear corner of the car ahead.
                if front_mid < 0.33:
                    command = "slow_down"
                else:
                    # If the model doesn't already request a lane change, force one.
                    if command not in ("left", "right"):
                        if prefer_left:
                            command = "left"
                        elif prefer_right:
                            command = "right"
                        else:
                            command = "slow_down"
        else:
            # Default (local Driving project) sequence-based format.
            sequence = self.vehicle_state.get_sequence_data()

            # Reshape for model: (1, sequence_length, num_features)
            sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])

            # Normalize
            original_shape = sequence.shape
            sequence_flat = sequence.reshape(-1, sequence.shape[-1])
            sequence_scaled = scaler.transform(sequence_flat) if scaler is not None else sequence_flat
            sequence = sequence_scaled.reshape(original_shape)

            prediction_probs = model(sequence, training=False).numpy()
            predicted_index = int(np.argmax(prediction_probs))
            command = _decode_action(encoder, predicted_index)
        
        # Get FSM recommendation
        if fsm_controller:
            fsm_action = fsm_controller.get_recommended_action()
            
            # Safety check
            if safety_analyzer:
                # Prefer FSM action if safer
                fsm_ttc = safety_analyzer.action_ttc.get(fsm_action, 0)
                ai_ttc = safety_analyzer.action_ttc.get(command, 0)
                
                if fsm_ttc > ai_ttc * 1.2:  # FSM significantly safer
                    command = fsm_action

        # Final safety override (prevents last-second crashes).
        # If chosen action is unsafe, pick the safest available action by TTC/prob.
        if safety_analyzer is not None:
            ttc = safety_analyzer.action_ttc.get(command, float('inf'))
            prob = safety_analyzer.action_collision_prob.get(command, 0.0)

            is_safe = (ttc > TTC_CRITICAL and prob < COLLISION_PROBABILITY_THRESHOLD)
            if not is_safe:
                # Prefer safe actions with highest TTC; else least collision probability.
                actions = ['forward', 'left', 'right', 'slow_down']
                safe = [
                    a for a in actions
                    if (safety_analyzer.action_ttc.get(a, float('inf')) > TTC_CRITICAL)
                    and (safety_analyzer.action_collision_prob.get(a, 1.0) < COLLISION_PROBABILITY_THRESHOLD)
                ]
                if safe:
                    command = max(safe, key=lambda a: safety_analyzer.action_ttc.get(a, 0.0))
                else:
                    command = min(actions, key=lambda a: safety_analyzer.action_collision_prob.get(a, 1.0))

        return command
    
    def _manual_mode_control(self):
        """Manual keyboard control"""
        keys = pygame.key.get_pressed()
        self.controls['forward'] = keys[pygame.K_UP] or keys[pygame.K_w]
        self.controls['reverse'] = keys[pygame.K_DOWN] or keys[pygame.K_s]
        self.controls['left'] = keys[pygame.K_LEFT] or keys[pygame.K_a]
        self.controls['right'] = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    
    def _apply_action(self, action):
        """Apply an action command"""
        if action == "forward":
            self.controls['forward'] = True
        elif action == "left":
            self.controls['forward'] = True
            self.controls['left'] = True
            if self.speed > 3.5:
                self.speed -= 0.1
        elif action == "right":
            self.controls['forward'] = True
            self.controls['right'] = True
            if self.speed > 3.5:
                self.speed -= 0.1
        elif action == "slow_down":
            self.controls['reverse'] = True
    
    def _apply_physics(self):
        """Apply physics and collision detection"""
        # Speed control
        if self.controls['forward']:
            if self.speed < MAX_SPEED:
                self.speed += ACCELERATION
        elif self.controls['reverse']:
            if self.speed > -MAX_SPEED:
                self.speed -= ACCELERATION
        else:
            # Deceleration
            if self.speed > 0:
                self.speed -= DECELERATION
            elif self.speed < 0:
                self.speed += DECELERATION
            if abs(self.speed) < DECELERATION:
                self.speed = 0
        
        if self.controls['left']:
            if self.controls['reverse']:
                self.angle -= ROTATION_SPEED
            else:
                self.angle += ROTATION_SPEED
        elif self.controls['right']:
            if self.controls['reverse']:
                self.angle += ROTATION_SPEED
            else:
                self.angle -= ROTATION_SPEED
        else:
            # AUTO-CENTER: Gradually return to 0 degrees
            if abs(self.angle) < 0.5:
                self.angle = 0
            elif self.angle > 0:
                self.angle -= ROTATION_SPEED * 0.3
            else:
                self.angle += ROTATION_SPEED * 0.3
        
        # Clamp angle
        self.angle = max(-MAX_ROTATION_ANGLE, min(self.angle, MAX_ROTATION_ANGLE))

        
        # Calculate movement
        radians = math.radians(self.angle)
        dx = -math.sin(radians) * self.speed
        dy = -math.cos(radians) * self.speed
        
        # Check collision
        if self.check_collision(self.x + dx, self.y + dy):
            return False
        
        # Apply movement
        self.x += dx
        self.y += dy
        
        # Boundary check
        road_x = (WIDTH - ROAD_WIDTH) // 2
        if self.x < road_x:
            self.x = road_x
        elif self.x + CAR_WIDTH > road_x + ROAD_WIDTH:
            self.x = road_x + ROAD_WIDTH - CAR_WIDTH
        
        return True
    
    def _get_current_lane(self):
        """Get current lane index"""
        road_left_edge = (WIDTH - ROAD_WIDTH) // 2
        lane_width = ROAD_WIDTH / LANE_COUNT
        center_x = self.x + CAR_WIDTH / 2
        lane = int((center_x - road_left_edge) / lane_width)
        return max(0, min(lane, LANE_COUNT - 1))
    
    def _get_current_action(self):
        """Get current action string"""
        if self.controls['left']:
            return "left"
        elif self.controls['right']:
            return "right"
        elif self.controls['reverse']:
            return "slow_down"
        else:
            return "forward"
    
    def _update_target_lane(self):
        """Update target lane for teacher mode"""
        road_left_edge = (WIDTH - ROAD_WIDTH) // 2
        lane_width = ROAD_WIDTH / LANE_COUNT
        
        center_x = self.x + CAR_WIDTH / 2
        current_physical_lane = int((center_x - road_left_edge) / lane_width)
        current_physical_lane = max(0, min(current_physical_lane, LANE_COUNT - 1))
        
        if current_physical_lane != self.target_lane:
            return
        
        # Check forward distance
        closest_dist = 9999
        for car in self.traffic.cars:
            if car['lane'] == self.target_lane and car['rect'].y < self.y:
                dist = self.y - car['rect'].bottom
                if dist < closest_dist:
                    closest_dist = dist
        
        if closest_dist < 300:
            # Check lanes
            left_safe = self._is_lane_safe(self.target_lane - 1)
            right_safe = self._is_lane_safe(self.target_lane + 1)
            
            if left_safe:
                self.target_lane -= 1
                self.lane_change_cooldown = 60
            elif right_safe:
                self.target_lane += 1
                self.lane_change_cooldown = 60
    
    def _is_lane_safe(self, lane):
        """Check if lane is safe for change"""
        if lane < 0 or lane >= LANE_COUNT:
            return False
        
        for car in self.traffic.cars:
            if car['lane'] == lane:
                if abs(car['rect'].y - self.y) < 350:
                    return False
        return True
    
    def _execute_lane_centering(self):
        """Execute lane centering logic"""
        road_left_edge = (WIDTH - ROAD_WIDTH) // 2
        lane_width = ROAD_WIDTH / LANE_COUNT
        target_x = road_left_edge + (self.target_lane * lane_width) + (lane_width / 2) - (CAR_WIDTH / 2)
        error_x = target_x - self.x
        
        if abs(error_x) < 10:
            if self.angle > 2:
                return "right"
            if self.angle < -2:
                return "left"
            
            # Check forward
            closest_dist = 9999
            for car in self.traffic.cars:
                if car['lane'] == self.target_lane and car['rect'].y < self.y:
                    dist = self.y - car['rect'].bottom
                    if dist < closest_dist:
                        closest_dist = dist
            
            if closest_dist < 200:
                return "slow_down"
            return "forward"
        
        if error_x < 0:
            if self.angle > 25:
                return "forward"
            return "left"
        else:
            if self.angle < -25:
                return "forward"
            return "right"
    
    def check_collision(self, new_x, new_y):
        """Check for collisions"""
        road_x = (WIDTH - ROAD_WIDTH) // 2
        if new_x < road_x or new_x + CAR_WIDTH > road_x + ROAD_WIDTH:
            return True
        
        player_rect = pygame.Rect(
            new_x + 10, new_y + 10,
            CAR_WIDTH - 20, self.car_height - 20
        )
        
        for car in self.traffic.cars:
            if car['rect'].colliderect(player_rect):
                if not car.get('stopped', False):
                    car['speed'] = 0
                    car['stopped'] = True
                    car['stop_time'] = time.time()
                return True
            
            if car.get('stopped', False):
                if time.time() - car['stop_time'] > 5:
                    car['speed'] = 2
                    car['stopped'] = False
        
        return False
    
    def draw(self, screen):
        """Draw car and sensors"""
        self.sensor.draw(screen)
        
        rotated_image = pygame.transform.rotate(self.image, self.angle)
        rotated_rect = rotated_image.get_rect(
            center=(self.x + CAR_WIDTH // 2, self.y + self.car_height // 2)
        )
        screen.blit(rotated_image, rotated_rect.topleft)