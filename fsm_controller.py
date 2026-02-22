"""
Finite State Machine Controller
Manages high-level driving behavior states
"""

from config import FSM_STATES, TTC_CRITICAL, TTC_WARNING
from enum import Enum


class DrivingState(Enum):
    LANE_KEEPING = "LANE_KEEPING"
    LANE_CHANGE_LEFT = "LANE_CHANGE_LEFT"
    LANE_CHANGE_RIGHT = "LANE_CHANGE_RIGHT"
    EMERGENCY_BRAKE = "EMERGENCY_BRAKE"
    ADAPTIVE_CRUISE = "ADAPTIVE_CRUISE"


class FSMController:
    """Finite State Machine for autonomous driving decisions"""
    
    def __init__(self):
        self.current_state = DrivingState.LANE_KEEPING
        self.previous_state = None
        self.state_entry_time = 0
        self.state_duration = 0
        self.lane_change_cooldown = 0
        
    def update(self, ego_state, surrounding_vehicles, safety_analysis, frame_count):
        """Update FSM state based on current situation"""
        self.state_duration = frame_count - self.state_entry_time
        
        if self.lane_change_cooldown > 0:
            self.lane_change_cooldown -= 1
        
        # Get safety metrics
        forward_ttc = safety_analysis.action_ttc['forward']
        left_safety = safety_analysis.action_ttc['left']
        right_safety = safety_analysis.action_ttc['right']
        
        forward_prob = safety_analysis.action_collision_prob['forward']
        history_risk = safety_analysis.get_history_window_risk()
        
        # STATE TRANSITIONS
        
        # 1. EMERGENCY BRAKE - Highest priority
        if forward_ttc < TTC_CRITICAL or forward_prob > 0.7:
            self._transition_to(DrivingState.EMERGENCY_BRAKE, frame_count)
        
        # 2. From EMERGENCY_BRAKE, exit when safe
        elif self.current_state == DrivingState.EMERGENCY_BRAKE:
            if forward_ttc > TTC_WARNING and forward_prob < 0.3:
                self._transition_to(DrivingState.ADAPTIVE_CRUISE, frame_count)
        
        # 3. From LANE_CHANGE states, complete the maneuver
        elif self.current_state == DrivingState.LANE_CHANGE_LEFT:
            target_lane = ego_state['lane'] - 1
            if target_lane < 0:
                self._transition_to(DrivingState.LANE_KEEPING, frame_count)
            elif self._is_lane_centered(ego_state, target_lane):
                self._transition_to(DrivingState.LANE_KEEPING, frame_count)
                self.lane_change_cooldown = 60
        
        elif self.current_state == DrivingState.LANE_CHANGE_RIGHT:
            target_lane = ego_state['lane'] + 1
            if target_lane >= 3:
                self._transition_to(DrivingState.LANE_KEEPING, frame_count)
            elif self._is_lane_centered(ego_state, target_lane):
                self._transition_to(DrivingState.LANE_KEEPING, frame_count)
                self.lane_change_cooldown = 60
        
        # 4. From ADAPTIVE_CRUISE, return to normal when safe
        elif self.current_state == DrivingState.ADAPTIVE_CRUISE:
            if forward_ttc > TTC_WARNING and forward_prob < 0.2:
                self._transition_to(DrivingState.LANE_KEEPING, frame_count)
        
        # 5. From LANE_KEEPING, decide if lane change needed
        elif self.current_state == DrivingState.LANE_KEEPING:
            # Check if blocked ahead
            if forward_ttc < TTC_WARNING or forward_prob > 0.4:
                # Consider lane change if not in cooldown
                if self.lane_change_cooldown == 0:
                    # Check which lane change is safer
                    if left_safety > right_safety and left_safety > TTC_WARNING:
                        self._transition_to(DrivingState.LANE_CHANGE_LEFT, frame_count)
                    elif right_safety > TTC_WARNING:
                        self._transition_to(DrivingState.LANE_CHANGE_RIGHT, frame_count)
                    else:
                        # Can't change lanes, slow down
                        self._transition_to(DrivingState.ADAPTIVE_CRUISE, frame_count)
                else:
                    # In cooldown, just slow down
                    self._transition_to(DrivingState.ADAPTIVE_CRUISE, frame_count)
    
    def _transition_to(self, new_state, frame_count):
        """Transition to new state"""
        if new_state != self.current_state:
            self.previous_state = self.current_state
            self.current_state = new_state
            self.state_entry_time = frame_count
            self.state_duration = 0
    
    def _is_lane_centered(self, ego_state, target_lane):
        """Check if vehicle is centered in target lane"""
        from config import WIDTH, ROAD_WIDTH, LANE_WIDTH, CAR_WIDTH
        
        road_left = (WIDTH - ROAD_WIDTH) // 2
        target_x = road_left + (target_lane * LANE_WIDTH) + (LANE_WIDTH / 2) - (CAR_WIDTH / 2)
        error = abs(ego_state['position']['x'] - target_x)
        
        return error < 15  # Within 15 pixels of center
    
    def get_recommended_action(self):
        """Get recommended action based on current state"""
        if self.current_state == DrivingState.EMERGENCY_BRAKE:
            return "slow_down"
        elif self.current_state == DrivingState.LANE_CHANGE_LEFT:
            return "left"
        elif self.current_state == DrivingState.LANE_CHANGE_RIGHT:
            return "right"
        elif self.current_state == DrivingState.ADAPTIVE_CRUISE:
            return "forward"  # Will be adjusted by speed control
        else:  # LANE_KEEPING
            return "forward"
    
    def get_state_name(self):
        """Get current state as string"""
        return self.current_state.value