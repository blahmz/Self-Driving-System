"""
Safety Analyzer — TTC & collision probability
Adapted for two-lane bidirectional road (lane 0 same-dir, lane 1 oncoming).
"""

import numpy as np
from collections import deque
from config import (
    TTC_CRITICAL, TTC_WARNING, COLLISION_PROBABILITY_THRESHOLD,
    SEQUENCE_LENGTH, TIME_PER_FRAME, SAFETY_DISTANCE_THRESHOLD,
    LANE_WIDTH, WIDTH, ROAD_WIDTH, CAR_WIDTH,
    ONCOMING_CLEAR_AHEAD, ONCOMING_CLEAR_BEHIND,
    OVERTAKE_TRIGGER_GAP,
)


class SafetyAnalyzer:
    """Action-specific TTC and collision probability calculations."""

    def __init__(self):
        self.ttc_history            = deque(maxlen=SEQUENCE_LENGTH)
        self.collision_prob_history = deque(maxlen=SEQUENCE_LENGTH)

        self.action_ttc = {
            'forward':   float('inf'),
            'left':      float('inf'),
            'right':     float('inf'),
            'slow_down': float('inf'),
        }
        self.action_collision_prob = {
            'forward':   0.0,
            'left':      0.0,
            'right':     0.0,
            'slow_down': 0.0,
        }

    # ------------------------------------------------------------------ #
    def calculate_ttc(self, ego_state, target_vehicle):
        """Longitudinal TTC (ego chasing car ahead or oncoming)."""
        if target_vehicle is None:
            return float('inf')

        rel_y   = ego_state['position']['y'] - target_vehicle['position']['y']
        ego_vy  = ego_state['velocity']['y']
        tgt_vy  = target_vehicle.get('velocity', 0)
        rel_vel = ego_vy - tgt_vy

        if rel_vel >= 0:
            return float('inf')

        ttc = -rel_y / rel_vel if rel_vel != 0 else float('inf')
        return max(0.0, ttc)

    # ------------------------------------------------------------------ #
    def calculate_collision_probability(self, ttc, relative_distance):
        if ttc == float('inf'):
            return 0.0

        if ttc < TTC_CRITICAL:
            ttc_prob = 1.0 - ttc / TTC_CRITICAL
        elif ttc < TTC_WARNING:
            ttc_prob = 0.5 * (1.0 - (ttc - TTC_CRITICAL) / (TTC_WARNING - TTC_CRITICAL))
        else:
            ttc_prob = 0.0

        dist_prob = max(0.0, 1.0 - relative_distance / SAFETY_DISTANCE_THRESHOLD) \
                    if relative_distance < SAFETY_DISTANCE_THRESHOLD else 0.0

        return min(1.0, 0.7 * ttc_prob + 0.3 * dist_prob)

    # ------------------------------------------------------------------ #
    def analyze_action_safety(self, action, ego_state, surrounding_vehicles):
        ego_lane = ego_state['lane']
        ego_y    = ego_state['position']['y']

        if action == 'forward':
            ahead = surrounding_vehicles.get_closest_vehicle_ahead(ego_lane, ego_y)
            ttc   = self.calculate_ttc(ego_state, ahead)
            if ahead:
                rel_dist = abs(ego_y - ahead['position']['y'])
                prob     = self.calculate_collision_probability(ttc, rel_dist)
            else:
                prob = 0.0

        elif action == 'left':
            # Lane 0 is leftmost — can't go further left
            if ego_lane == 0:
                return 0.0, 1.0, False
            # Moving from lane 1 to lane 0
            target_lane = 0
            vehicles = surrounding_vehicles.get_vehicles_by_lane(target_lane)
            ttcs, probs = [], []
            for v in vehicles:
                t    = self.calculate_ttc(ego_state, v)
                rdist = abs(ego_y - v['position']['y'])
                p    = self.calculate_collision_probability(t, rdist)
                ttcs.append(t); probs.append(p)
            ttc  = min(ttcs)  if ttcs  else float('inf')
            prob = max(probs) if probs else 0.0

        elif action == 'right':
            # Moving from lane 0 to lane 1 (oncoming lane)
            if ego_lane == 1:
                return 0.0, 1.0, False
            # Check oncoming cars in lane 1
            right_clear = surrounding_vehicles.is_lane_clear_for_overtake(
                ego_y, ONCOMING_CLEAR_AHEAD, ONCOMING_CLEAR_BEHIND)
            if not right_clear:
                return 0.0, 0.9, False
            ttc  = float('inf')
            prob = 0.0

        elif action == 'slow_down':
            behind = surrounding_vehicles.get_closest_vehicle_behind(ego_lane, ego_y)
            if behind and abs(behind['position']['y'] - ego_y) < 100:
                ttc  = 1.0
                prob = 0.3
            else:
                ttc  = float('inf')
                prob = 0.0
        else:
            ttc  = float('inf')
            prob = 0.0

        is_safe = ttc > TTC_CRITICAL and prob < COLLISION_PROBABILITY_THRESHOLD
        return ttc, prob, is_safe

    # ------------------------------------------------------------------ #
    def analyze_all_actions(self, ego_state, surrounding_vehicles):
        results = {}
        for action in ['forward', 'left', 'right', 'slow_down']:
            ttc, prob, is_safe = self.analyze_action_safety(action, ego_state, surrounding_vehicles)
            results[action] = {'ttc': ttc, 'collision_prob': prob, 'is_safe': is_safe}
            self.action_ttc[action]            = ttc
            self.action_collision_prob[action] = prob

        min_ttc  = min(r['ttc']            for r in results.values())
        max_prob = max(r['collision_prob']  for r in results.values())
        self.ttc_history.append(min_ttc)
        self.collision_prob_history.append(max_prob)
        return results

    # ------------------------------------------------------------------ #
    def get_history_window_risk(self):
        if len(self.ttc_history) < SEQUENCE_LENGTH:
            return 0.0
        ttcs  = list(self.ttc_history)
        probs = list(self.collision_prob_history)
        avg_prob = np.mean(probs)
        ttc_trend  = (ttcs[-1]  - ttcs[0])  / len(ttcs)
        prob_trend = (probs[-1] - probs[0]) / len(probs)
        risk = avg_prob
        if ttc_trend  < 0: risk += 0.2
        if prob_trend > 0: risk += 0.2
        return min(1.0, risk)

    # ------------------------------------------------------------------ #
    def get_safest_action(self, results):
        safe = {a: d for a, d in results.items() if d['is_safe']}
        if not safe:
            return min(results, key=lambda a: results[a]['collision_prob'])
        return max(safe, key=lambda a: safe[a]['ttc'])
