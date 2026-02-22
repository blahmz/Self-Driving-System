"""
Advanced HUD Display System
Shows vehicle state, safety metrics, FSM state, and traffic density
"""

import pygame
from config import (WHITE, BLACK, CYAN, YELLOW, RED, GREEN, ORANGE, 
                    TTC_CRITICAL, TTC_WARNING)


class HUDDisplay:
    """Comprehensive HUD for autonomous driving visualization"""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.font_large = pygame.font.SysFont("Arial", 28, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 22)
        self.font_small = pygame.font.SysFont("Arial", 18)
        
    def draw_main_panel(self, screen, mode_info, stats):
        """Draw main information panel"""
        panel_width = 320
        panel_height = 240
        x, y = 10, 10
        
        # Semi-transparent background
        s = pygame.Surface((panel_width, panel_height))
        s.set_alpha(200)
        s.fill(BLACK)
        screen.blit(s, (x, y))
        pygame.draw.rect(screen, WHITE, (x, y, panel_width, panel_height), 2)
        
        # Title
        title = self.font_large.render("AUTONOMOUS SYSTEM", True, CYAN)
        screen.blit(title, (x + 10, y + 10))
        
        # Mode indicator
        mode_color = CYAN if mode_info['is_teacher'] else (GREEN if mode_info['is_ai'] else YELLOW)
        mode_text = self.font_medium.render(f"Mode: {mode_info['mode_name']}", True, mode_color)
        screen.blit(mode_text, (x + 10, y + 45))
        
        # Stats
        y_offset = 75
        stats_list = [
            f"Frames: {stats['frames']}",
            f"Distance: {stats['distance']:.0f} m",
            f"Overtakes: {stats['overtakes']}",
            f"Speed: {stats['speed']:.1f} km/h",
            f"Lane: {stats['lane'] + 1}/3"
        ]
# Add auto-run timer if enabled
        if hasattr(stats, 'auto_run_time_remaining'):
            time_remaining = stats['auto_run_time_remaining']
            if time_remaining is not None:
                timer_color = GREEN if time_remaining > 60 else (ORANGE if time_remaining > 30 else RED)
                timer_text = self.font_small.render(
                    f"Auto-run: {int(time_remaining)}s left", 
                    True, timer_color
                )
                screen.blit(timer_text, (x + 10, y + y_offset + 25))
        
        for stat in stats_list:
            text = self.font_small.render(stat, True, WHITE)
            screen.blit(text, (x + 10, y + y_offset))
            y_offset += 25
        
        # Controls hint
        hint = self.font_small.render("T=Teach A=AI M=Manual", True, GRAY := (150, 150, 150))
        screen.blit(hint, (x + 10, y + 210))
    
    def draw_vehicle_state_panel(self, screen, vehicle_state):
        """Draw detailed vehicle state"""
        panel_width = 320
        panel_height = 200
        x = 10
        y = 260
        
        # Background
        s = pygame.Surface((panel_width, panel_height))
        s.set_alpha(200)
        s.fill(BLACK)
        screen.blit(s, (x, y))
        pygame.draw.rect(screen, CYAN, (x, y, panel_width, panel_height), 2)
        
        # Title
        title = self.font_medium.render("VEHICLE STATE", True, CYAN)
        screen.blit(title, (x + 10, y + 10))
        
        # State details
        state = vehicle_state.get_state_vector()
        y_offset = 45
        
        state_lines = [
            f"Position: ({state['position']['x']:.0f}, {state['position']['y']:.0f})",
            f"Velocity: {state['speed']:.2f} px/s",
            f"Acceleration: {state['acceleration']['y']:.2f}",
            f"Angle: {state['angle']:.1f}°",
            f"Target Lane: {state.get('target_lane', 'N/A')}"
        ]
        
        for line in state_lines:
            text = self.font_small.render(line, True, WHITE)
            screen.blit(text, (x + 10, y + y_offset))
            y_offset += 25
    
    def draw_safety_panel(self, screen, safety_analysis):
        """Draw safety metrics and TTC"""
        panel_width = 380
        panel_height = 280
        x = self.width - panel_width - 10
        y = 10
        
        # Background
        s = pygame.Surface((panel_width, panel_height))
        s.set_alpha(200)
        s.fill(BLACK)
        screen.blit(s, (x, y))
        pygame.draw.rect(screen, RED, (x, y, panel_width, panel_height), 2)
        
        # Title
        title = self.font_medium.render("SAFETY ANALYSIS", True, RED)
        screen.blit(title, (x + 10, y + 10))
        
        # Action-specific TTC
        y_offset = 50
        actions = ['forward', 'left', 'right', 'slow_down']
        
        for action in actions:
            ttc = safety_analysis.action_ttc[action]
            prob = safety_analysis.action_collision_prob[action]
            
            # Color based on safety
            if ttc < TTC_CRITICAL or prob > 0.5:
                color = RED
                status = "DANGER"
            elif ttc < TTC_WARNING or prob > 0.3:
                color = ORANGE
                status = "WARNING"
            else:
                color = GREEN
                status = "SAFE"
            
            # Action name
            action_text = self.font_small.render(
                f"{action.upper()[:4]}:", True, WHITE
            )
            screen.blit(action_text, (x + 10, y + y_offset))
            
            # TTC
            if ttc == float('inf'):
                ttc_str = "∞"
            else:
                ttc_str = f"{ttc:.2f}s"
            
            ttc_text = self.font_small.render(ttc_str, True, color)
            screen.blit(ttc_text, (x + 100, y + y_offset))
            
            # Probability
            prob_text = self.font_small.render(f"{prob*100:.1f}%", True, color)
            screen.blit(prob_text, (x + 200, y + y_offset))
            
            # Status
            status_text = self.font_small.render(status, True, color)
            screen.blit(status_text, (x + 280, y + y_offset))
            
            y_offset += 30
        
        # History risk
        history_risk = safety_analysis.get_history_window_risk()
        risk_color = RED if history_risk > 0.5 else (ORANGE if history_risk > 0.3 else GREEN)
        
        risk_label = self.font_small.render("History Risk:", True, WHITE)
        screen.blit(risk_label, (x + 10, y + y_offset + 10))
        
        risk_value = self.font_medium.render(f"{history_risk*100:.1f}%", True, risk_color)
        screen.blit(risk_value, (x + 150, y + y_offset + 10))
        
        # Draw risk bar
        bar_width = 200
        bar_height = 20
        bar_x = x + 150
        bar_y = y + y_offset + 45
        
        pygame.draw.rect(screen, WHITE, (bar_x, bar_y, bar_width, bar_height), 1)
        filled_width = int(bar_width * min(1.0, history_risk))
        pygame.draw.rect(screen, risk_color, (bar_x, bar_y, filled_width, bar_height))
    
    def draw_fsm_state(self, screen, fsm_controller):
        """Draw FSM state indicator"""
        panel_width = 380
        panel_height = 100
        x = self.width - panel_width - 10
        y = 300
        
        # Background
        s = pygame.Surface((panel_width, panel_height))
        s.set_alpha(200)
        s.fill(BLACK)
        screen.blit(s, (x, y))
        pygame.draw.rect(screen, YELLOW, (x, y, panel_width, panel_height), 2)
        
        # Title
        title = self.font_medium.render("FSM STATE", True, YELLOW)
        screen.blit(title, (x + 10, y + 10))
        
        # Current state (large)
        state_name = fsm_controller.get_state_name()
        state_text = self.font_large.render(state_name, True, CYAN)
        screen.blit(state_text, (x + 10, y + 45))
        
        # Duration
        duration_text = self.font_small.render(
            f"Duration: {fsm_controller.state_duration} frames", True, WHITE
        )
        screen.blit(duration_text, (x + 10, y + 75))
    
    def draw_surrounding_vehicles_panel(self, screen, surrounding_vehicles):
        """Draw surrounding vehicle count and details"""
        panel_width = 380
        panel_height = 120
        x = self.width - panel_width - 10
        y = 410
        
        # Background
        s = pygame.Surface((panel_width, panel_height))
        s.set_alpha(200)
        s.fill(BLACK)
        screen.blit(s, (x, y))
        pygame.draw.rect(screen, CYAN, (x, y, panel_width, panel_height), 2)
        
        # Title
        title = self.font_medium.render("SURROUNDING VEHICLES", True, CYAN)
        screen.blit(title, (x + 10, y + 10))
        
        # Total count
        total = surrounding_vehicles.get_vehicle_count()
        count_text = self.font_large.render(f"Total: {total}", True, GREEN if total < 5 else ORANGE)
        screen.blit(count_text, (x + 10, y + 45))
        
        # Per lane breakdown
        y_offset = 85
        for lane in range(3):
            lane_vehicles = surrounding_vehicles.get_vehicles_by_lane(lane)
            count = len(lane_vehicles)
            
            lane_text = self.font_small.render(
                f"Lane {lane + 1}: {count}", True, WHITE
            )
            screen.blit(lane_text, (x + 10 + lane * 120, y + y_offset))
    
    def draw_density_control(self, screen, density_manager):
        """Draw traffic density control panel"""
        panel_width = 380
        panel_height = 100
        x = self.width - panel_width - 10
        y = 540
        
        # Background
        s = pygame.Surface((panel_width, panel_height))
        s.set_alpha(200)
        s.fill(BLACK)
        screen.blit(s, (x, y))
        pygame.draw.rect(screen, GREEN, (x, y, panel_width, panel_height), 2)
        
        # Title
        title = self.font_medium.render("TRAFFIC DENSITY", True, GREEN)
        screen.blit(title, (x + 10, y + 10))
        
        # Current density
        density_text = self.font_large.render(
            density_manager.current_density, True, YELLOW
        )
        screen.blit(density_text, (x + 10, y + 45))
        
        # Config
        config = density_manager.get_spawn_parameters()
        config_text = self.font_small.render(
            f"Target: {config['num_cars']} cars | Gap: {config['min_gap']}px",
            True, WHITE
        )
        screen.blit(config_text, (x + 10, y + 75))
        
        # Controls
        hint = self.font_small.render("[ = decrease  ] = increase", True, (150, 150, 150))
        screen.blit(hint, (x + 200, y + 45))