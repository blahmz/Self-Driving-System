"""
Main Simulation Entry Point
"""
import pygame, os, csv, sys, time, pickle
import numpy as np

from config import *
from car_controller  import Car, load_image
from road_traffic    import Road, Traffic
from vehicle_state   import VehicleState, SurroundingVehicles
from safety_analyzer import SafetyAnalyzer
from fsm_controller  import FSMController
from hud_display     import HUDDisplay

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

class Simulation:
    def __init__(self):
        pygame.init()
        
        # FIXED: Resizable window
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Autonomous Driving v2.1 (F11=Fullscreen)")
        self.clock  = pygame.time.Clock()

        # Components
        self.road    = Road()
        self.traffic = Traffic()
        self.car     = Car(self.road, self.traffic)

        self.surrounding_vehicles = SurroundingVehicles()
        self.safety_analyzer      = SafetyAnalyzer()
        self.fsm_controller       = FSMController()
        self.hud                  = HUDDisplay(WIDTH, HEIGHT)

        self.ai_model = self.ai_encoder = self.ai_scaler = None
        self._load_ai()

        # State
        self.teacher_mode = False # Start in Manual
        self.ai_mode      = False
        
        self.road_offset_1 = 0.0
        self.road_offset_2 = float(-HEIGHT)
        self.current_run_data  = []
        self.distance_traveled = 0.0
        self.total_overtakes   = 0
        self.frame_count       = 0
        self.running           = True
        
        self.is_fullscreen = False
        self.zoom_out  = True   # Default zoomed out to show more road and vehicles
        self.zoom_scale = 0.42  # Proportional scale: more road visible, sprites stay sharp

        print("\nCONTROLS:")
        print(" T - Teacher Mode (autonomous with FSM)")
        print(" A - AI Mode (LSTM neural net)")
        print(" M - Manual Mode")
        print(" Z - Toggle Zoom Out (see oncoming traffic earlier)")
        print(" [ / ] - Fewer / More same-dir cars")
        print(" F11 - Toggle Fullscreen")
        print(" ESC - Quit")

    def _load_ai(self):
        if not AI_AVAILABLE: return
        if os.path.exists(MODEL_FILE):
            try:
                self.ai_model   = load_model(MODEL_FILE)
                with open(LABEL_ENCODER_FILE, 'rb') as f:
                    self.ai_encoder = pickle.load(f)
                with open(SCALER_FILE, 'rb') as f:
                    self.ai_scaler  = pickle.load(f)
                print(f"[AI] Model loaded.")
            except Exception as e:
                print(f"[AI] Load error: {e}")

    def handle_events(self):
        import config as cfg
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k == pygame.K_ESCAPE:
                    self.running = False
                elif k == pygame.K_z:
                    self.zoom_out = not self.zoom_out
                    print(f"[VIEW] Zoom {'OUT' if self.zoom_out else 'NORMAL'}")
                elif k == pygame.K_F11: # Toggle Fullscreen
                    self._toggle_fullscreen()
                elif k == pygame.K_t:
                    self.teacher_mode = True;  self.ai_mode = False
                    print("[MODE] Teacher Active")
                elif k == pygame.K_a:
                    if self.ai_model:
                        self.teacher_mode = False; self.ai_mode = True
                        print("[MODE] AI Active")
                    else:
                        print("[MODE] AI Unavailable")
                elif k == pygame.K_m:
                    self.teacher_mode = False; self.ai_mode = False
                    print("[MODE] Manual Active")
                elif k == pygame.K_LEFTBRACKET:
                    cfg.SAME_DIR_NUM_CARS = max(1, cfg.SAME_DIR_NUM_CARS - 1)
                elif k == pygame.K_RIGHTBRACKET:
                    cfg.SAME_DIR_NUM_CARS = min(8, cfg.SAME_DIR_NUM_CARS + 1)
            
            elif event.type == pygame.VIDEORESIZE:
                # Handle window resize
                if not self.is_fullscreen:
                    new_w, new_h = event.w, event.h
                    self.screen = pygame.display.set_mode((new_w, new_h), pygame.RESIZABLE)
                    self.hud.width = new_w
                    self.hud.height = new_h

    def _toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            info = pygame.display.Info()
            self.hud.width = info.current_w
            self.hud.height = info.current_h
        else:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            self.hud.width = WIDTH
            self.hud.height = HEIGHT

    def update(self):
        self.frame_count += 1
        scroll = self.car.speed

        self.road_offset_1 += scroll
        self.road_offset_2 += scroll
        if self.road_offset_1 >= HEIGHT: self.road_offset_1 = self.road_offset_2 - HEIGHT
        if self.road_offset_2 >= HEIGHT: self.road_offset_2 = self.road_offset_1 - HEIGHT

        self.surrounding_vehicles.update(self.traffic.cars, {'x': self.car.x, 'y': self.car.y})
        self.car.sensor.update(self.road.borders, self.traffic)
        
        state = self.car.vehicle_state.get_state_vector()
        self.safety_analyzer.analyze_all_actions(state, self.surrounding_vehicles)
        self.fsm_controller.update(state, self.surrounding_vehicles, self.safety_analyzer, self.frame_count)

        result = self.car.move(
            is_ai=self.ai_mode, model=self.ai_model,
            encoder=self.ai_encoder, scaler=self.ai_scaler,
            is_teacher=self.teacher_mode,
            fsm_controller=self.fsm_controller,
            safety_analyzer=self.safety_analyzer,
            skip_sensor_update=True
        )

        if not result:
            self._crash_screen()
            self.running = False
            return

        self.car.y = float(HEIGHT - self.car.car_height - 120)

        new_ot = self.traffic.update(scroll, self.car.y)
        self.total_overtakes += new_ot
        self.distance_traveled += self.car.speed * 0.05

        if not self.ai_mode:
            self._collect_data()

    def render(self):
        sw = self.screen.get_width()
        sh = self.screen.get_height()

        if self.zoom_out:
            # Render full scene to offscreen surface, then scale down so
            # the player can see oncoming traffic much earlier.
            surf = pygame.Surface((WIDTH, HEIGHT))
            surf.fill(BLACK)
            self.road.draw(surf, self.road_offset_1, self.road_offset_2)
            self.traffic.draw(surf)
            self.car.draw(surf)

            scaled_w = int(WIDTH * self.zoom_scale)
            scaled_h = int(HEIGHT * self.zoom_scale)
            scaled   = pygame.transform.smoothscale(surf, (scaled_w, scaled_h))
            ox = (sw - scaled_w) // 2
            oy = (sh - scaled_h) // 2

            self.screen.fill(BLACK)
            self.screen.blit(scaled, (ox, oy))

            font = pygame.font.SysFont("Consolas", 14)
            lbl  = font.render("ZOOM OUT  [Z] to toggle", True, (120, 120, 120))
            self.screen.blit(lbl, (ox + 4, oy + scaled_h - 20))
        else:
            self.screen.fill(BLACK)
            self.road.draw(self.screen, self.road_offset_1, self.road_offset_2)
            self.traffic.draw(self.screen)
            self.car.draw(self.screen)

        self._draw_hud()
        pygame.display.flip()

    def _draw_hud(self):
        mode_name = ("TEACHER" if self.teacher_mode else "AI" if self.ai_mode else "MANUAL")
        mode_info = {'mode_name': mode_name, 'is_teacher': self.teacher_mode, 'is_ai': self.ai_mode}
        stats = {
            'frames': self.frame_count, 'distance': self.distance_traveled,
            'overtakes': self.total_overtakes, 'speed': self.car.speed * 21.6,
            'lane': self.car._get_current_lane(), 'tgt_speed': self.car.target_speed * 21.6,
        }
        self.hud.draw_main_panel(self.screen, mode_info, stats)
        self.hud.draw_vehicle_state_panel(self.screen, self.car.vehicle_state)
        self.hud.draw_safety_panel(self.screen, self.safety_analyzer)
        self.hud.draw_fsm_state(self.screen, self.fsm_controller)
        self.hud.draw_surrounding_vehicles_panel(self.screen, self.surrounding_vehicles)
        self.hud.draw_lane_diagram(self.screen, self.car._get_current_lane(), self.traffic.cars)
        self._draw_oncoming_gap_indicator()

    def _collect_data(self):
        if not self.car.sensor.readings: return
        row = [r['offset'] for r in self.car.sensor.readings]
        a = self.car._get_current_action()
        row.append(a)
        self.current_run_data.append(row)

    def _crash_screen(self):
        font = pygame.font.SysFont("Impact", 90)
        msg  = font.render("COLLISION!", True, RED)
        self.screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - 60))
        pygame.display.flip()
        pygame.time.wait(2000)

    def _save_data(self):
        if not self.current_run_data: return
        exists = os.path.isfile(TRAINING_DATA_FILE)
        with open(TRAINING_DATA_FILE, 'a', newline='') as f:
            w = csv.writer(f)
            if not exists:
                 n = len(self.current_run_data[0]) - 1
                 w.writerow([f"sensor_{i+1}" for i in range(n)] + ["decision"])
            w.writerows(self.current_run_data)
        print(f"Saved {len(self.current_run_data)} samples.")

    def _draw_oncoming_gap_indicator(self):
        """
        Bottom-centre bar showing whether the right (oncoming) lane is clear.
        GREEN = clear window open (safe to overtake)
        RED   = oncoming car present or approaching
        Also shows countdown in frames during quiet window.
        """
        cooldown = self.traffic.get_oncoming_cooldown()
        onc_on   = any(c['going_down'] for c in self.traffic.cars)

        sw = self.screen.get_width()
        bw, bh = 320, 28
        bx = (sw - bw) // 2
        by = self.screen.get_height() - bh - 6

        # Background
        s = pygame.Surface((bw, bh))
        s.set_alpha(200)
        s.fill((0, 0, 0))
        self.screen.blit(s, (bx, by))

        if onc_on:
            bar_col = (220, 30, 30)
            label   = "ONCOMING — RIGHT LANE BLOCKED"
        elif cooldown > 0:
            frac    = cooldown / self.traffic.MAX_QUIET_FRAMES
            bar_col = (50, 200, 50)
            secs    = cooldown // 60
            label   = f"RIGHT LANE CLEAR  ~{secs}s window"
        else:
            bar_col = (255, 140, 0)
            label   = "ONCOMING INCOMING SOON"

        # Filled bar
        pygame.draw.rect(self.screen, bar_col, (bx, by, bw, bh), border_radius=4)
        pygame.draw.rect(self.screen, (255, 255, 255), (bx, by, bw, bh), 2, border_radius=4)

        font = pygame.font.SysFont("Consolas", 13, bold=True)
        txt  = font.render(label, True, (0, 0, 0))
        self.screen.blit(txt, (bx + (bw - txt.get_width()) // 2,
                               by + (bh - txt.get_height()) // 2))

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(FPS)
        pygame.quit()
        self._save_data()

if __name__ == "__main__":
    Simulation().run()