"""
Ego Car Controller
"""
import pygame
import math
import os
import numpy as np

from config import (
    VEHICLE_SCALE, CAR_WIDTH, MAX_ROTATION_ANGLE, ROTATION_SPEED,
    MAX_SPEED, CRUISE_SPEED, OVERTAKE_SPEED, MIN_FOLLOW_SPEED,
    ACCELERATION, DECELERATION, BRAKE_FORCE, FOLLOW_SPEED_MARGIN,
    WIDTH, ROAD_WIDTH, LANE_COUNT, LANE_WIDTH, HEIGHT,
    DECISION_FREQUENCY, ROAD_LEFT, ROAD_RIGHT, LANE_CENTERS,
    OVERTAKE_TRIGGER_GAP, OVERTAKE_COMMIT_GAP,
    ONCOMING_CLEAR_AHEAD, ONCOMING_CLEAR_BEHIND,
    EMERGENCY_BRAKE_DIST, _CAR_H_REF, TTC_WARNING,
)
from sensors import Sensor
from vehicle_state import VehicleState

def load_image(name, fallback_color=(220, 200, 0), fallback_size=(52, 90)):
    for path in [name, name.replace('.jpg', '.png'), name.replace('.png', '.jpg')]:
        if os.path.exists(path):
            return pygame.image.load(path).convert_alpha()
    surf = pygame.Surface(fallback_size, pygame.SRCALPHA)
    surf.fill(fallback_color)
    return surf

class Car:
    def __init__(self, road, traffic):
        # ── Image: Scale strictly to config ──
        raw = load_image('car.png', (240, 210, 40), (52, 90))
        rw, rh = raw.get_size()
        
        # FIXED: Use CAR_WIDTH from config as absolute authority
        target_w = CAR_WIDTH
        target_h = int(rh * (target_w / rw)) 
        
        self.car_height = target_h
        self.image = pygame.transform.scale(raw, (target_w, target_h))

        # Starting position
        self.x = float(LANE_CENTERS[0] - CAR_WIDTH // 2)
        self.y = float(HEIGHT - self.car_height - 120)

        self.speed        = 1.0
        self.target_speed = CRUISE_SPEED
        self.angle        = 0.0

        self.controls = {'forward': False, 'reverse': False,
                         'left': False, 'right': False}

        self.road    = road
        self.traffic = traffic

        self.sensor        = Sensor(self)
        self.vehicle_state = VehicleState()

        self.target_lane          = 0
        self.lane_change_cooldown = 0
        self.decision_counter     = 0
        self.last_ai_command      = "forward"

        self._blocker_id   = None
        self._blocker_bottom_y = None
        self._overtake_wish_frames = 0  # Frames we've wanted to overtake (avoid flicker)

    # ─────────────────────────────────────────────────────────────────────
    def move(self, is_ai=False, model=None, encoder=None, scaler=None,
             is_teacher=False, fsm_controller=None, safety_analyzer=None,
             skip_sensor_update=False):
        
        self.controls = {k: False for k in self.controls}
        self.decision_counter += 1

        current_lane    = self._get_current_lane()
        sensor_readings = [r['offset'] for r in self.sensor.readings]
        action          = self._get_current_action()
        self.vehicle_state.update(
            self.x, self.y, self.angle, self.speed,
            current_lane, sensor_readings, action
        )

        if is_teacher:
            self._teacher_mode(safety_analyzer=safety_analyzer)
        elif is_ai:
            if model is not None and encoder is not None:
                if self.decision_counter % DECISION_FREQUENCY == 0:
                    self.last_ai_command = self._ai_mode(
                        model, encoder, scaler, fsm_controller, safety_analyzer)
            self._apply_action(self.last_ai_command)
        else:
            self._manual_mode()

        if not self._apply_physics():
            return False

        if not skip_sensor_update:
            self.sensor.update(self.road.borders, self.traffic)
        return True

    # ── TEACHER MODE (Logic Based) ───────────────────────────────────────
    def _teacher_mode(self, safety_analyzer=None):
        if self.lane_change_cooldown > 0:
            self.lane_change_cooldown -= 1
        else:
            self._teacher_decide(safety_analyzer=safety_analyzer)

        # Execute lane centering logic (steering)
        action = self._execute_lane_centering()
        self._apply_action(action)

    def _teacher_decide(self, safety_analyzer=None):
        current = self._get_current_lane()
        if current != self.target_lane:
            return   # mid-lane-change: commit, do not oscillate

        if self.target_lane == 0:
            # LEFT LANE: Check for blockers and forward TTC in current lane
            gap, blocker = self._gap_and_blocker_in_lane(0)
            fwd_ttc = float('inf')
            if safety_analyzer is not None:
                fwd_ttc = safety_analyzer.action_ttc.get('forward', float('inf'))

            if gap < OVERTAKE_TRIGGER_GAP:
                # Slow down immediately if too close
                if gap < EMERGENCY_BRAKE_DIST:
                    self.target_speed = MIN_FOLLOW_SPEED
                else:
                    blk_spd = blocker['speed'] if blocker else MIN_FOLLOW_SPEED
                    self.target_speed = max(MIN_FOLLOW_SPEED, blk_spd - FOLLOW_SPEED_MARGIN)

                # Overtake only when: tailing (forward TTC in current lane), safe target-lane gaps, commit after stable wish
                target_lane_safe = self._right_lane_clear()
                tailing = fwd_ttc < TTC_WARNING and fwd_ttc != float('inf')
                if gap < OVERTAKE_TRIGGER_GAP and target_lane_safe:
                    self._overtake_wish_frames = min(60, self._overtake_wish_frames + 1)
                else:
                    self._overtake_wish_frames = 0

                if (gap < OVERTAKE_COMMIT_GAP and target_lane_safe and tailing
                        and self._overtake_wish_frames >= 5):
                    if blocker:
                        self._blocker_id = blocker['id']
                        self._blocker_bottom_y = blocker['rect'].bottom
                    self.target_lane = 1
                    self.lane_change_cooldown = 60
                    self._overtake_wish_frames = 0
            else:
                self.target_speed = CRUISE_SPEED
                self._overtake_wish_frames = 0

        elif self.target_lane == 1:
            # RIGHT LANE: Overtaking
            self.target_speed = OVERTAKE_SPEED

            # ABORT: if oncoming car has appeared, return to left lane immediately (safety first)
            if not self._right_lane_clear():
                self.target_lane = 0
                self.lane_change_cooldown = 180
                self._blocker_id = None
                self._overtake_wish_frames = 0
                self.target_speed = MIN_FOLLOW_SPEED
                return

            # Check if we passed the blocker
            if self._blocker_is_behind():
                # Ensure left lane is safe to merge back
                if self._left_lane_safe_to_merge():
                    self.target_lane = 0
                    self.lane_change_cooldown = 60
                    self._blocker_id = None

    def _left_lane_safe_to_merge(self):
        """Check if there is a car in the left lane immediately beside us."""
        for car in self.traffic.cars:
            if car['lane'] == 0 and not car['going_down']:
                dist = abs(car['rect'].centery - (self.y + self.car_height//2))
                if dist < _CAR_H_REF * 1.2: # Too close beside
                    return False
        return True

    def _gap_and_blocker_in_lane(self, lane):
        best_gap = float('inf')
        best_car = None
        ego_front = self.y
        for car in self.traffic.cars:
            if car['lane'] == lane and not car['going_down']:
                car_bottom = car['rect'].bottom
                if car_bottom < ego_front: # Ahead
                    gap = ego_front - car_bottom
                    if gap < best_gap:
                        best_gap = gap
                        best_car = car
        return best_gap, best_car

    def _right_lane_clear(self):
        """
        Senior Logic: Check oncoming lane.
        We need clearance AHEAD and BEHIND.
        """
        for car in self.traffic.cars:
            if car['going_down']: # Oncoming
                # Relative position
                cy = car['rect'].centery
                # Is it in our danger zone?
                # Zone extends far ahead because of closing speed
                if (cy < self.y + ONCOMING_CLEAR_AHEAD and
                    cy > self.y - ONCOMING_CLEAR_BEHIND):
                    return False
        return True

    def _blocker_is_behind(self):
        if self._blocker_id:
            for car in self.traffic.cars:
                if car['id'] == self._blocker_id:
                    # Passed if blocker's top is below our bottom
                    return car['rect'].top > self.y + self.car_height + 20
            return True # Blocker disappeared
        return False

    # ── LANE CENTERING (Steering Control) ───────────────────────────────
    def _execute_lane_centering(self):
        target_x = ROAD_LEFT + self.target_lane * LANE_WIDTH + \
                   LANE_WIDTH // 2 - CAR_WIDTH // 2
        error_x  = target_x - self.x

        # P-Controller for steering
        # If error is large, steer hard. If small, fine tune.
        steer_threshold = 2.0
        
        if error_x < -steer_threshold:
            return "left"
        elif error_x > steer_threshold:
            return "right"
        else:
            # Straighten out if close to center
            if self.angle > 2: return "right"
            if self.angle < -2: return "left"
            
            # Speed control handled in decide, but fallback
            if self.target_speed < self.speed - 0.2:
                return "slow_down"
            return "forward"

    # ── AI MODE ─────────────────────────────────────────────────────────
    def _ai_mode(self, model, encoder, scaler, fsm_controller, safety_analyzer):
        nn_action = "forward"
        try:
            nn_action = self._get_nn_prediction(model, encoder, scaler)
        except Exception:
            pass

        fsm_action = fsm_controller.get_recommended_action() if fsm_controller else "forward"

        # Priority Logic
        if fsm_action == "slow_down":
            return fsm_action

        # FSM State updates (Synchronized with Car)
        if fsm_controller:
            state = fsm_controller.current_state.value

            # ABORT: oncoming appeared mid-overtake — go back left NOW
            if state == 'ABORT_OVERTAKE' and self.target_lane != 0:
                self.target_lane = 0
                self.lane_change_cooldown = 180
                self._blocker_id = None
                self.target_speed = MIN_FOLLOW_SPEED

            elif state in ('OVERTAKE_OUT', 'OVERTAKE_PASS') and self.target_lane != 1:
                # FSM says go right, check safety first
                if safety_analyzer.action_collision_prob['right'] < 0.2:
                    gap, blocker = self._gap_and_blocker_in_lane(0)
                    self._blocker_id = blocker['id'] if blocker else None
                    self.target_lane = 1
                    self.lane_change_cooldown = 60

            elif state in ('OVERTAKE_IN', 'LANE_KEEPING') and self.target_lane != 0:
                if self._blocker_is_behind() and self._left_lane_safe_to_merge():
                    self.target_lane = 0
                    self.lane_change_cooldown = 60
                    self._blocker_id = None

        # Fallback to NN if FSM is idle
        if nn_action == "right" and self.target_lane == 0:
             if safety_analyzer.action_collision_prob['right'] < 0.15:
                self.target_lane = 1
                self.lane_change_cooldown = 60

        return self._execute_lane_centering()

    def _get_nn_prediction(self, model, encoder, scaler):
        seq = self.vehicle_state.get_sequence_data()
        if seq is None or len(seq) == 0: return "forward"
        n_steps, n_feat = seq.shape
        x = seq.reshape(1, n_steps, n_feat).astype(np.float32)
        if scaler is not None:
            try:
                x = scaler.transform(x.reshape(-1, n_feat)).reshape(1, n_steps, n_feat)
            except: pass
        pred = model.predict(x, verbose=0)
        idx  = int(np.argmax(pred))
        if hasattr(encoder, 'inverse_transform'):
            return encoder.inverse_transform([idx])[0]
        return "forward"

    # ── MANUAL ──────────────────────────────────────────────────────────
    def _manual_mode(self):
        keys = pygame.key.get_pressed()
        self.controls['forward'] = keys[pygame.K_UP]    or keys[pygame.K_w]
        self.controls['reverse'] = keys[pygame.K_DOWN]  or keys[pygame.K_s]
        self.controls['left']    = keys[pygame.K_LEFT]  or keys[pygame.K_a]
        self.controls['right']   = keys[pygame.K_RIGHT] or keys[pygame.K_d]

    def _apply_action(self, action):
        if action == "forward":
            self.controls['forward'] = True
        elif action == "left":
            self.controls['forward'] = True
            self.controls['left']    = True
        elif action == "right":
            self.controls['forward'] = True
            self.controls['right']   = True
        elif action == "slow_down":
            self.controls['reverse'] = True

    # ── PHYSICS ─────────────────────────────────────────────────────────
    def _apply_physics(self):
        # Speed Physics
        if self.controls['reverse']:
            self.speed -= BRAKE_FORCE
        elif self.controls['forward']:
            if self.speed < self.target_speed:
                self.speed += ACCELERATION
            elif self.speed > self.target_speed:
                self.speed -= DECELERATION
        else:
            self.speed -= DECELERATION

        self.speed = max(0.0, min(self.speed, MAX_SPEED))

        # Steering Physics
        if self.controls['left']:
            self.angle += ROTATION_SPEED
        elif self.controls['right']:
            self.angle -= ROTATION_SPEED
        else:
            # Auto-center
            if abs(self.angle) < 0.5: self.angle = 0.0
            elif self.angle > 0: self.angle -= ROTATION_SPEED * 0.4
            else: self.angle += ROTATION_SPEED * 0.4

        self.angle = max(-MAX_ROTATION_ANGLE, min(self.angle, MAX_ROTATION_ANGLE))

        # Movement
        rad = math.radians(self.angle)
        dx  = -math.sin(rad) * self.speed
        dy  = -math.cos(rad) * self.speed # Slight forward drift based on angle

        if self._check_collision(self.x + dx, self.y + dy):
            self.speed *= 0.2
            return False

        self.x = max(float(ROAD_LEFT), min(self.x + dx, float(ROAD_RIGHT - CAR_WIDTH)))
        self.y += dy
        return True

    def _cos(self, deg):
        return math.cos(math.radians(deg))

    def _check_collision(self, nx, ny):
        # Scaled padding for smaller car
        pad = int(CAR_WIDTH * 0.1) 
        ego_rect = pygame.Rect(
            int(nx) + pad, int(ny) + pad,
            CAR_WIDTH - 2 * pad, self.car_height - 2 * pad
        )
        for car in self.traffic.cars:
            tc = pygame.Rect(
                car['rect'].x + pad, car['rect'].y + pad,
                car['rect'].w - 2 * pad, car['rect'].h - 2 * pad
            )
            if ego_rect.colliderect(tc):
                return True
        return False

    def check_collision(self, nx, ny):
        return self._check_collision(nx, ny)

    def _get_current_lane(self):
        cx = self.x + CAR_WIDTH / 2
        return max(0, min(int((cx - ROAD_LEFT) / LANE_WIDTH), LANE_COUNT - 1))

    def _get_current_action(self):
        if self.controls['left']:    return "left"
        if self.controls['right']:   return "right"
        if self.controls['reverse']: return "slow_down"
        return "forward"

    def draw(self, screen):
        self.sensor.draw(screen)
        rotated = pygame.transform.rotate(self.image, self.angle)
        rect    = rotated.get_rect(
            center=(int(self.x + CAR_WIDTH // 2),
                    int(self.y + self.car_height // 2)))
        screen.blit(rotated, rect.topleft)