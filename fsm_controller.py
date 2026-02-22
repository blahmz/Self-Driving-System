"""
FSM Controller — Two-lane bidirectional overtake scenario

States:
  LANE_KEEPING    : cruise in left lane at CRUISE_SPEED
  ADAPTIVE_CRUISE : following slow car, speed-matched
  OVERTAKE_OUT    : moving right to pass blocker
  OVERTAKE_PASS   : in right lane, accelerating past blocker
  OVERTAKE_IN     : returning to left lane
  ABORT_OVERTAKE  : oncoming car appeared mid-overtake — abort back to left lane
  EMERGENCY_BRAKE : hard brake, imminent collision
"""

from enum import Enum
from config import (
    TTC_CRITICAL, TTC_WARNING,
    OVERTAKE_TRIGGER_GAP, OVERTAKE_COMMIT_GAP,
    ONCOMING_CLEAR_AHEAD, ONCOMING_CLEAR_BEHIND,
    OVERTAKE_COMMIT_MIN_FRAMES,
    WIDTH, ROAD_WIDTH, LANE_WIDTH, CAR_WIDTH, _CAR_H_REF,
)


class DrivingState(Enum):
    LANE_KEEPING    = "LANE_KEEPING"
    ADAPTIVE_CRUISE = "ADAPTIVE_CRUISE"
    OVERTAKE_OUT    = "OVERTAKE_OUT"
    OVERTAKE_PASS   = "OVERTAKE_PASS"
    OVERTAKE_IN     = "OVERTAKE_IN"
    ABORT_OVERTAKE  = "ABORT_OVERTAKE"
    EMERGENCY_BRAKE = "EMERGENCY_BRAKE"


class FSMController:
    def __init__(self):
        self.current_state      = DrivingState.LANE_KEEPING
        self.previous_state     = None
        self.state_entry_frame  = 0
        self.state_duration     = 0
        self.lane_change_cooldown = 0
        self._blocker_bottom_y  = None
        self._overtake_commit_until_frame = 0  # Don't leave OVERTAKE_OUT until committed

    # ── main update ────────────────────────────────────────────────────
    def update(self, ego_state, surrounding_vehicles, safety_analysis, frame):
        self.state_duration = frame - self.state_entry_frame
        if self.lane_change_cooldown > 0:
            self.lane_change_cooldown -= 1

        ego_lane  = ego_state['lane']
        ego_y     = ego_state['position']['y']
        fwd_ttc   = safety_analysis.action_ttc['forward']
        fwd_prob  = safety_analysis.action_collision_prob['forward']

        # Right lane clearance check — used throughout
        right_clear = surrounding_vehicles.is_lane_clear_for_overtake(
            ego_y, ONCOMING_CLEAR_AHEAD, ONCOMING_CLEAR_BEHIND)

        gap_ahead = surrounding_vehicles.get_gap_to_car_ahead(
            ego_lane, ego_y, _CAR_H_REF)

        # ── Priority 0: EMERGENCY_BRAKE ──────────────────────────────
        if fwd_ttc < TTC_CRITICAL or fwd_prob > 0.75:
            self._transition(DrivingState.EMERGENCY_BRAKE, frame)

        # ── Exit EMERGENCY_BRAKE ─────────────────────────────────────
        elif self.current_state == DrivingState.EMERGENCY_BRAKE:
            if fwd_ttc > TTC_WARNING and fwd_prob < 0.3:
                self._transition(DrivingState.LANE_KEEPING, frame)

        # ── ABORT_OVERTAKE: oncoming appeared while we were out ──────
        elif self.current_state == DrivingState.ABORT_OVERTAKE:
            # Just steer back to left lane — handled by car controller
            if ego_lane == 0 and self._in_lane_centre(ego_state, 0):
                self.lane_change_cooldown = 180   # 3-second cooldown after abort
                self._transition(DrivingState.ADAPTIVE_CRUISE, frame)
            elif self.state_duration > 240:
                # Safety timeout — should never happen but prevents freezing
                self._transition(DrivingState.LANE_KEEPING, frame)

        # ── OVERTAKE_OUT: steering right (commit: no oscillate) ────────
        elif self.current_state == DrivingState.OVERTAKE_OUT:
            if not right_clear:
                self._transition(DrivingState.ABORT_OVERTAKE, frame)
            elif ego_lane == 1 and self._in_lane_centre(ego_state, 1):
                self._transition(DrivingState.OVERTAKE_PASS, frame)
            elif self.state_duration > 150:
                self._transition(DrivingState.ABORT_OVERTAKE, frame)
            # Else stay in OVERTAKE_OUT until we reach lane 1 or timeout (commit)

        # ── OVERTAKE_PASS: in right lane ─────────────────────────────
        elif self.current_state == DrivingState.OVERTAKE_PASS:
            if not right_clear:
                # Oncoming car arrived — move back immediately
                self._transition(DrivingState.ABORT_OVERTAKE, frame)
            else:
                blocker = surrounding_vehicles.get_closest_vehicle_ahead(0, ego_y)
                # Passed when blocker is behind (its y > ego bottom)
                passed = (blocker is None or
                          blocker['rect'].top > ego_y + _CAR_H_REF + 50)
                if passed:
                    self._transition(DrivingState.OVERTAKE_IN, frame)
                elif self.state_duration > 200:   # timeout safety
                    self._transition(DrivingState.OVERTAKE_IN, frame)

        # ── OVERTAKE_IN: returning left ──────────────────────────────
        elif self.current_state == DrivingState.OVERTAKE_IN:
            if ego_lane == 0 and self._in_lane_centre(ego_state, 0):
                self.lane_change_cooldown = 120  # 2-second cooldown
                if gap_ahead < OVERTAKE_TRIGGER_GAP:
                    self._transition(DrivingState.ADAPTIVE_CRUISE, frame)
                else:
                    self._transition(DrivingState.LANE_KEEPING, frame)
            elif self.state_duration > 200:
                self._transition(DrivingState.LANE_KEEPING, frame)

        # ── ADAPTIVE_CRUISE: following ───────────────────────────────
        elif self.current_state == DrivingState.ADAPTIVE_CRUISE:
            if fwd_ttc > TTC_WARNING and fwd_prob < 0.2:
                self._transition(DrivingState.LANE_KEEPING, frame)
            # Start overtake only when tailing (forward TTC) and target lane has safe front/rear gaps
            elif (gap_ahead < OVERTAKE_COMMIT_GAP
                  and fwd_ttc < TTC_WARNING
                  and right_clear
                  and self.lane_change_cooldown == 0):
                blocker = surrounding_vehicles.get_closest_vehicle_ahead(0, ego_y)
                self._blocker_bottom_y = blocker['rect'].bottom if blocker else None
                self._overtake_commit_until_frame = frame + OVERTAKE_COMMIT_MIN_FRAMES
                self._transition(DrivingState.OVERTAKE_OUT, frame)

        # ── LANE_KEEPING: normal cruise ──────────────────────────────
        elif self.current_state == DrivingState.LANE_KEEPING:
            if fwd_ttc < TTC_WARNING or fwd_prob > 0.4:
                self._transition(DrivingState.ADAPTIVE_CRUISE, frame)
            # Overtake only with proper forward TTC (tailing) and safe target-lane gaps
            if (gap_ahead < OVERTAKE_TRIGGER_GAP
                    and fwd_ttc < TTC_WARNING
                    and right_clear
                    and self.lane_change_cooldown == 0):
                blocker = surrounding_vehicles.get_closest_vehicle_ahead(0, ego_y)
                self._blocker_bottom_y = blocker['rect'].bottom if blocker else None
                self._overtake_commit_until_frame = frame + OVERTAKE_COMMIT_MIN_FRAMES
                self._transition(DrivingState.OVERTAKE_OUT, frame)

    # ── helpers ────────────────────────────────────────────────────────
    def _transition(self, new_state, frame):
        if new_state != self.current_state:
            self.previous_state    = self.current_state
            self.current_state     = new_state
            self.state_entry_frame = frame
            self.state_duration    = 0

    def _in_lane_centre(self, ego_state, lane):
        from config import ROAD_LEFT
        target_x = ROAD_LEFT + lane * LANE_WIDTH + LANE_WIDTH // 2 - CAR_WIDTH // 2
        return abs(ego_state['position']['x'] - target_x) < 18

    def get_recommended_action(self):
        s = self.current_state
        if s == DrivingState.EMERGENCY_BRAKE:  return "slow_down"
        if s == DrivingState.OVERTAKE_OUT:     return "right"
        if s == DrivingState.OVERTAKE_PASS:    return "forward"
        if s == DrivingState.OVERTAKE_IN:      return "left"
        if s == DrivingState.ABORT_OVERTAKE:   return "left"   # abort = go back left
        if s == DrivingState.ADAPTIVE_CRUISE:  return "slow_down"
        return "forward"

    def get_state_name(self):
        return self.current_state.value
