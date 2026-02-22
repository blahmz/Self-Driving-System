"""
Safety Analyzer - TTC and Collision Probability Calculations
Non-Markovian decision making with history window
"""

import numpy as np
import math
from collections import deque
from config import (TTC_CRITICAL, TTC_WARNING, COLLISION_PROBABILITY_THRESHOLD,
                    SEQUENCE_LENGTH, TIME_PER_FRAME, SAFETY_DISTANCE_THRESHOLD,
                    LANE_WIDTH, WIDTH, ROAD_WIDTH, CAR_WIDTH)


class SafetyAnalyzer:
    """Analyzes safety constraints and collision risks"""
    
    def __init__(self):
        self.ttc_history = deque(maxlen=SEQUENCE_LENGTH)
        self.collision_prob_history = deque(maxlen=SEQUENCE_LENGTH)
        self.safety_violations = []
        
        # Action-specific TTC
        self.action_ttc = {
            'forward': float('inf'),
            'left': float('inf'),
            'right': float('inf'),
            'slow_down': float('inf')
        }
        
        self.action_collision_prob = {
            'forward': 0.0,
            'left': 0.0,
            'right': 0.0,
            'slow_down': 0.0
        }
    
    def calculate_ttc(self, ego_state, target_vehicle):
        """
        Calculate Time-To-Collision
        TTC = (relative_distance) / (relative_velocity)
        """
        if target_vehicle is None:
            return float('inf')
        
        # Relative position
        rel_y = ego_state['position']['y'] - target_vehicle['position']['y']
        
        # Relative velocity (negative means closing)
        ego_vel_y = ego_state['velocity']['y']
        target_vel_y = target_vehicle.get('velocity', 0)
        rel_vel = ego_vel_y - target_vel_y
        
        # If not approaching, no collision
        if rel_vel >= 0:
            return float('inf')
        
        # TTC calculation
        ttc = -rel_y / rel_vel if rel_vel != 0 else float('inf')
        
        return max(0, ttc)
    
    def calculate_lateral_ttc(self, ego_state, target_vehicle, target_lane):
        """Calculate TTC for lane change maneuver"""
        if target_vehicle is None:
            return float('inf')
        
        # Distance in y-direction
        rel_y = abs(ego_state['position']['y'] - target_vehicle['position']['y'])
        
        # Lateral separation
        road_left = (WIDTH - ROAD_WIDTH) // 2
        ego_lane_center = road_left + (ego_state['lane'] * LANE_WIDTH) + (LANE_WIDTH / 2)
        target_lane_center = road_left + (target_lane * LANE_WIDTH) + (LANE_WIDTH / 2)
        lateral_distance = abs(ego_lane_center - target_lane_center)
        
        # Estimate time to complete lane change
        lane_change_time = lateral_distance / max(abs(ego_state['velocity']['x']), 20)
        
        # During lane change, check if paths will intersect
        ego_vel_y = ego_state['velocity']['y']
        target_vel_y = target_vehicle.get('velocity', 0)
        
        # Relative velocity
        rel_vel = ego_vel_y - target_vel_y
        
        if rel_vel >= 0:
            return float('inf')
        
        # TTC during lane change
        ttc = rel_y / abs(rel_vel) if rel_vel != 0 else float('inf')
        
        # If lane change will complete before collision, safe
        if lane_change_time < ttc:
            return float('inf')
        
        return max(0, ttc)
    
    def calculate_collision_probability(self, ttc, relative_distance):
        """
        Calculate collision probability based on TTC and distance
        P(collision) = f(TTC, distance, uncertainty)
        """
        if ttc == float('inf'):
            return 0.0
        
        # Base probability from TTC
        if ttc < TTC_CRITICAL:
            ttc_prob = 1.0 - (ttc / TTC_CRITICAL)
        elif ttc < TTC_WARNING:
            ttc_prob = 0.5 * (1.0 - (ttc - TTC_CRITICAL) / (TTC_WARNING - TTC_CRITICAL))
        else:
            ttc_prob = 0.0
        
        # Distance factor
        if relative_distance < SAFETY_DISTANCE_THRESHOLD:
            dist_prob = 1.0 - (relative_distance / SAFETY_DISTANCE_THRESHOLD)
        else:
            dist_prob = 0.0
        
        # Combined probability (weighted average)
        collision_prob = 0.7 * ttc_prob + 0.3 * dist_prob
        
        return min(1.0, collision_prob)
    
    def analyze_action_safety(self, action, ego_state, surrounding_vehicles):
        """
        Analyze safety of a specific action
        Returns (TTC, collision_probability, is_safe)
        """
        ego_lane = ego_state['lane']
        ego_y = ego_state['position']['y']
        
        if action == 'forward':
            # Check vehicle ahead
            ahead = surrounding_vehicles.get_closest_vehicle_ahead(ego_lane, ego_y)
            ttc = self.calculate_ttc(ego_state, ahead)
            
            if ahead:
                rel_dist = ego_y - ahead['position']['y']
                collision_prob = self.calculate_collision_probability(ttc, rel_dist)
            else:
                collision_prob = 0.0
                
        elif action == 'left':
            target_lane = ego_lane - 1
            if target_lane < 0:
                return 0.0, 1.0, False
            
            # Check all vehicles in target lane
            lane_vehicles = surrounding_vehicles.get_vehicles_by_lane(target_lane)
            ttcs = []
            probs = []
            
            for vehicle in lane_vehicles:
                lat_ttc = self.calculate_lateral_ttc(ego_state, vehicle, target_lane)
                ttcs.append(lat_ttc)
                
                rel_dist = abs(ego_y - vehicle['position']['y'])
                prob = self.calculate_collision_probability(lat_ttc, rel_dist)
                probs.append(prob)
            
            ttc = min(ttcs) if ttcs else float('inf')
            collision_prob = max(probs) if probs else 0.0
            
        elif action == 'right':
            target_lane = ego_lane + 1
            if target_lane >= 3:  # Assuming 3 lanes
                return 0.0, 1.0, False
            
            lane_vehicles = surrounding_vehicles.get_vehicles_by_lane(target_lane)
            ttcs = []
            probs = []
            
            for vehicle in lane_vehicles:
                lat_ttc = self.calculate_lateral_ttc(ego_state, vehicle, target_lane)
                ttcs.append(lat_ttc)
                
                rel_dist = abs(ego_y - vehicle['position']['y'])
                prob = self.calculate_collision_probability(lat_ttc, rel_dist)
                probs.append(prob)
            
            ttc = min(ttcs) if ttcs else float('inf')
            collision_prob = max(probs) if probs else 0.0
            
        elif action == 'slow_down':
            # Slowing down generally safe, but check behind
            behind = surrounding_vehicles.get_closest_vehicle_behind(ego_lane, ego_y)
            
            if behind:
                # If vehicle behind is very close and fast, braking might be risky
                rel_dist = behind['position']['y'] - ego_y
                if rel_dist < 100:  # Very close
                    ttc = 1.0
                    collision_prob = 0.3
                else:
                    ttc = float('inf')
                    collision_prob = 0.0
            else:
                ttc = float('inf')
                collision_prob = 0.0
        else:
            ttc = float('inf')
            collision_prob = 0.0
        
        # Determine if action is safe
        is_safe = (ttc > TTC_CRITICAL and collision_prob < COLLISION_PROBABILITY_THRESHOLD)
        
        return ttc, collision_prob, is_safe
    
    def analyze_all_actions(self, ego_state, surrounding_vehicles):
        """Analyze safety of all possible actions"""
        results = {}
        
        for action in ['forward', 'left', 'right', 'slow_down']:
            ttc, prob, is_safe = self.analyze_action_safety(action, ego_state, surrounding_vehicles)
            
            results[action] = {
                'ttc': ttc,
                'collision_prob': prob,
                'is_safe': is_safe
            }
            
            # Store for quick access
            self.action_ttc[action] = ttc
            self.action_collision_prob[action] = prob
        
        # Store in history
        min_ttc = min([r['ttc'] for r in results.values()])
        max_prob = max([r['collision_prob'] for r in results.values()])
        
        self.ttc_history.append(min_ttc)
        self.collision_prob_history.append(max_prob)
        
        return results
    
    def get_safest_action(self, action_safety_results):
        """Return safest action based on analysis"""
        safe_actions = {action: data for action, data in action_safety_results.items() 
                       if data['is_safe']}
        
        if not safe_actions:
            # Emergency: all actions unsafe, choose least dangerous
            return min(action_safety_results.keys(), 
                      key=lambda a: action_safety_results[a]['collision_prob'])
        
        # Among safe actions, choose one with highest TTC
        return max(safe_actions.keys(), 
                  key=lambda a: safe_actions[a]['ttc'])
    
    def get_history_window_risk(self):
        """Calculate risk based on history window (non-Markovian)"""
        if len(self.ttc_history) < SEQUENCE_LENGTH:
            return 0.0
        
        # Calculate trend
        recent_ttcs = list(self.ttc_history)
        recent_probs = list(self.collision_prob_history)
        
        # Average risk
        avg_prob = np.mean(recent_probs)
        
        # Trend (getting worse?)
        if len(recent_ttcs) > 1:
            ttc_trend = (recent_ttcs[-1] - recent_ttcs[0]) / len(recent_ttcs)
            prob_trend = (recent_probs[-1] - recent_probs[0]) / len(recent_probs)
        else:
            ttc_trend = 0
            prob_trend = 0
        
        # Combined risk (higher if trend is negative)
        risk = avg_prob
        if ttc_trend < 0:  # TTC decreasing (bad)
            risk += 0.2
        if prob_trend > 0:  # Probability increasing (bad)
            risk += 0.2
        
        return min(1.0, risk)