"""
HUD Display — Two-Lane Bidirectional Simulation
Shows mode, stats, FSM state, safety analysis, and surrounding vehicles.
"""

import pygame
from config import (
    WHITE, BLACK, CYAN, YELLOW, RED, GREEN, ORANGE, GRAY,
    TTC_CRITICAL, TTC_WARNING,
    SAME_DIR_NUM_CARS, ONCOMING_NUM_CARS,
)

_GRAY = (150, 150, 150)

class HUDDisplay:
    def __init__(self, width, height):
        self.width  = width
        self.height = height
        self.font_large  = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 18)
        self.font_small  = pygame.font.SysFont("Consolas", 14)
        self.font_state  = pygame.font.SysFont("Consolas", 22, bold=True)
        
        # Define custom colors for this class
        self.BLUE_COL = (60, 120, 220)

    # ── panel helper ──────────────────────────────────────────────────
    def _panel(self, screen, x, y, w, h, border_col=WHITE):
        s = pygame.Surface((w, h))
        s.set_alpha(200)
        s.fill(BLACK)
        screen.blit(s, (x, y))
        pygame.draw.rect(screen, border_col, (x, y, w, h), 2)

    # ── main panel ────────────────────────────────────────────────────
    def draw_main_panel(self, screen, mode_info, stats):
        pw, ph = 300, 230
        x, y = 10, 10
        self._panel(screen, x, y, pw, ph, CYAN)

        title = self.font_large.render("AUTONOMOUS  SYSTEM", True, CYAN)
        screen.blit(title, (x + 8, y + 8))

        mc = CYAN if mode_info['is_teacher'] else (GREEN if mode_info['is_ai'] else YELLOW)
        mt = self.font_medium.render(f"Mode: {mode_info['mode_name']}", True, mc)
        screen.blit(mt, (x + 8, y + 40))

        items = [
            f"Frames   : {stats['frames']}",
            f"Distance : {stats['distance']:.0f} m",
            f"Overtakes: {stats['overtakes']}",
            f"Speed    : {stats['speed']:.1f} km/h",
            f"Lane     : {'LEFT' if stats['lane'] == 0 else 'RIGHT'}",
        ]
        yy = y + 70
        for line in items:
            t = self.font_small.render(line, True, WHITE)
            screen.blit(t, (x + 8, yy))
            yy += 24

        hint = self.font_small.render("T=Teacher  A=AI  M=Manual  ESC=Quit", True, _GRAY)
        screen.blit(hint, (x + 8, y + ph - 22))

    # ── vehicle state ─────────────────────────────────────────────────
    def draw_vehicle_state_panel(self, screen, vehicle_state):
        pw, ph = 300, 190
        x, y = 10, 250
        self._panel(screen, x, y, pw, ph, CYAN)

        title = self.font_medium.render("VEHICLE STATE", True, CYAN)
        screen.blit(title, (x + 8, y + 8))

        state = vehicle_state.get_state_vector()
        yy = y + 38
        lines = [
            f"Pos  : ({state['position']['x']:.0f}, {state['position']['y']:.0f})",
            f"Speed: {state['speed']:.2f} px/f",
            f"Accel: {state['acceleration']['y']:.2f}",
            f"Angle: {state['angle']:.1f} deg",
            f"Lane : {state['lane']}  Action: {state['action']}",
        ]
        for line in lines:
            t = self.font_small.render(line, True, WHITE)
            screen.blit(t, (x + 8, yy))
            yy += 26

    # ── safety panel ──────────────────────────────────────────────────
    def draw_safety_panel(self, screen, safety_analysis):
        pw, ph = 360, 270
        x = self.width - pw - 10
        y = 10
        self._panel(screen, x, y, pw, ph, RED)

        title = self.font_medium.render("SAFETY ANALYSIS", True, RED)
        screen.blit(title, (x + 8, y + 8))

        headers = self.font_small.render(
            f"{'ACTION':<10} {'TTC':>7} {'PROB%':>7} {'STATUS':>8}", True, _GRAY)
        screen.blit(headers, (x + 8, y + 36))

        yy = y + 56
        for action in ['forward', 'right', 'left', 'slow_down']:
            ttc  = safety_analysis.action_ttc[action]
            prob = safety_analysis.action_collision_prob[action]
            if ttc < TTC_CRITICAL or prob > 0.5:
                col, status = RED,    "DANGER "
            elif ttc < TTC_WARNING or prob > 0.3:
                col, status = ORANGE, "WARNING"
            else:
                col, status = GREEN,  "SAFE   "

            ttc_s = f"{ttc:.2f}s" if ttc < 999 else "  inf"
            row   = f"{action.upper():<10} {ttc_s:>7} {prob*100:>6.1f}% {status:>8}"
            t = self.font_small.render(row, True, col)
            screen.blit(t, (x + 8, yy))
            yy += 26

        # History risk bar
        risk     = safety_analysis.get_history_window_risk()
        risk_col = RED if risk > 0.5 else (ORANGE if risk > 0.3 else GREEN)
        rl = self.font_small.render(f"History Risk: {risk * 100:.1f}%", True, WHITE)
        screen.blit(rl, (x + 8, yy + 8))

        bx, by, bw, bh = x + 8, yy + 30, pw - 20, 16
        pygame.draw.rect(screen, WHITE,     (bx, by, bw, bh), 1)
        pygame.draw.rect(screen, risk_col,  (bx, by, int(bw * min(1.0, risk)), bh))

    # ── FSM state ─────────────────────────────────────────────────────
    def draw_fsm_state(self, screen, fsm_controller):
        pw, ph = 360, 90
        x = self.width - pw - 10
        y = 290
        self._panel(screen, x, y, pw, ph, YELLOW)

        title = self.font_medium.render("FSM STATE", True, YELLOW)
        screen.blit(title, (x + 8, y + 8))

        state_col_map = {
            'LANE_KEEPING':    CYAN,
            'OVERTAKE_OUT':    ORANGE,
            'OVERTAKE_PASS':   GREEN,
            'OVERTAKE_IN':     ORANGE,
            'ABORT_OVERTAKE':  RED,
            'EMERGENCY_BRAKE': RED,
            'ADAPTIVE_CRUISE': YELLOW,
        }
        sname = fsm_controller.get_state_name()
        scol  = state_col_map.get(sname, WHITE)
        st    = self.font_state.render(sname, True, scol)
        screen.blit(st, (x + 8, y + 36))

        dur = self.font_small.render(f"Duration: {fsm_controller.state_duration} frames", True, _GRAY)
        screen.blit(dur, (x + 220, y + 44))

    # ── surrounding vehicles ──────────────────────────────────────────
    def draw_surrounding_vehicles_panel(self, screen, surrounding_vehicles):
        pw, ph = 360, 110
        x = self.width - pw - 10
        y = 390
        self._panel(screen, x, y, pw, ph, CYAN)

        title = self.font_medium.render("SURROUNDING VEHICLES", True, CYAN)
        screen.blit(title, (x + 8, y + 8))

        total = surrounding_vehicles.get_vehicle_count()
        lane0 = len(surrounding_vehicles.get_vehicles_by_lane(0))
        lane1 = len(surrounding_vehicles.get_vehicles_by_lane(1))

        t1 = self.font_medium.render(f"Total: {total}", True,
                                     GREEN if total < 6 else ORANGE)
        screen.blit(t1, (x + 8, y + 38))

        t2 = self.font_small.render(
            f"Left (same-dir): {lane0}   Right (oncoming): {lane1}", True, WHITE)
        screen.blit(t2, (x + 8, y + 68))

        t3 = self.font_small.render(
            "[ = less same-dir    ] = more same-dir", True, _GRAY)
        screen.blit(t3, (x + 8, y + 88))

    # ── lane diagram ──────────────────────────────────────────────────
    def draw_lane_diagram(self, screen, ego_lane, traffic_cars):
        """Mini top-view bird's-eye diagram bottom-right."""
        dw, dh = 160, 110
        x = self.width - dw - 10
        y = self.height - dh - 10
        self._panel(screen, x, y, dw, dh, (80, 80, 80))

        # Road outline
        road_x = x + 20
        mid_x  = x + 20 + (dw - 40) // 2
        road_r = x + dw - 20
        pygame.draw.rect(screen, (60, 60, 60), (road_x, y + 10, dw - 40, dh - 20))
        # Centre yellow line
        pygame.draw.line(screen, YELLOW, (mid_x, y + 10), (mid_x, y + dh - 10), 2)
        # Borders
        pygame.draw.line(screen, WHITE, (road_x, y + 10), (road_x, y + dh - 10), 2)
        pygame.draw.line(screen, WHITE, (road_r, y + 10), (road_r, y + dh - 10), 2)

        # Ego car marker
        ex = road_x + (dw - 40) // 4 - 5 if ego_lane == 0 else \
             road_x + 3 * (dw - 40) // 4 - 5
        pygame.draw.rect(screen, YELLOW, (ex, y + dh // 2 - 8, 10, 14), border_radius=2)

        # Traffic dots
        for car in traffic_cars:
            lx = road_x + (dw - 40) // 4 - 3 if car['lane'] == 0 else \
                 road_x + 3 * (dw - 40) // 4 - 3
            # Map car y (screen) to diagram y
            car_norm_y = max(0.0, min(1.0, car['rect'].centery / 720))
            car_dy = int(y + 10 + car_norm_y * (dh - 20))
            
            # FIXED: Used self.BLUE_COL defined in __init__
            col = ORANGE if not car['going_down'] else self.BLUE_COL
            pygame.draw.rect(screen, col, (lx, car_dy - 4, 6, 8), border_radius=1)

        label = self.font_small.render("TOP VIEW", True, _GRAY)
        screen.blit(label, (x + (dw - label.get_width()) // 2, y + 2))