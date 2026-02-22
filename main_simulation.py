"""
MAIN SIMULATION - Advanced Autonomous Driving System
Integrates LSTM, FSM, Safety Analysis, and Dynamic Traffic Density
"""

import pygame
import os
import csv
import time
import pickle
import numpy as np

# Import all modules
from config import *
from car_controller import Car, load_image
from road_traffic import Road, Traffic
from vehicle_state import VehicleState, SurroundingVehicles
from safety_analyzer import SafetyAnalyzer
from fsm_controller import FSMController
from density_manager import DensityManager
from datetime import datetime
from hud_display import HUDDisplay

# AI Model imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    AI_AVAILABLE = True
except ImportError:
    print("WARNING: TensorFlow not found. AI mode disabled.")
    AI_AVAILABLE = False


class AdvancedSimulation:
    """Main simulation controller"""
    
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
        pygame.display.set_caption("Advanced Autonomous Driving - LSTM + FSM + Safety")
        self.clock = pygame.time.Clock()
        
        # Load textures
        self.grass_texture = load_image("grass.jpg")
        self.grass_texture = pygame.transform.scale(self.grass_texture, (WIDTH, HEIGHT))
        
        # Initialize managers
        self.density_manager = DensityManager()
        
        # Initialize world
        self.road = Road(WIDTH // 2, ROAD_WIDTH, LANE_COUNT, self.grass_texture)
        self.traffic = Traffic(
            num_cars=self.density_manager.get_spawn_parameters()['num_cars'],
            lane_width=LANE_WIDTH,
            num_lanes=LANE_COUNT,
            road_width=ROAD_WIDTH,
            road=self.road,
            density_manager=self.density_manager
        )
        
        # Initialize car
        self.car = Car(self.road, self.traffic)
        
        # Initialize state trackers
        self.surrounding_vehicles = SurroundingVehicles()
        self.safety_analyzer = SafetyAnalyzer()
        self.fsm_controller = FSMController()
        
        # Initialize HUD
        self.hud = HUDDisplay(WIDTH, HEIGHT)
        
        # AI setup
        self.ai_model = None
        self.ai_encoder = None
        self.ai_scaler = None
        self.load_ai_model()
        
        # Mode control
        self.teacher_mode = True  # Start in teacher mode
        self.ai_mode = False
        
        # Simulation state
        self.road_offset_1 = 0
        self.road_offset_2 = -HEIGHT
        self.current_run_data = []
        self.distance_traveled = 0
        self.total_overtakes = 0
        self.frame_count = 0
# Auto-run tracking
        self.auto_run_enabled = AUTO_RUN_ENABLED
        self.auto_run_start_time = None
        self.auto_run_schedule_index = 0
        
        if self.auto_run_enabled:
            print("\n[AUTO] AUTO-RUN MODE ENABLED")
            print(f"Total Duration: {AUTO_RUN_TOTAL_TIME} seconds")
            print("Density Schedule:")
            for i, schedule in enumerate(AUTO_RUN_DENSITY_SCHEDULE):
                print(f"  {i+1}. {schedule['density']} for {schedule['duration']}s")
            print("="*60 + "\n")
        self.running = True
        
        print("\n" + "="*60)
        print("SIMULATION INITIALIZED")
        print("="*60)
        print(f"Mode: {'TEACHER' if self.teacher_mode else 'MANUAL'}")
        print(f"AI Available: {AI_AVAILABLE and self.ai_model is not None}")
        print(f"Traffic Density: {self.density_manager.current_density}")
        print("\nControls:")
        print("  T = Teacher Mode (Sequential Lane Change)")
        print("  A = AI Mode (LSTM + FSM)")
        print("  M = Manual Mode")
        print("  [ = Decrease Traffic Density")
        print("  ] = Increase Traffic Density")
        print("  ESC = Exit")
        print("="*60 + "\n")
    
    def load_ai_model(self):
        """Load trained AI model and artifacts"""
        if not AI_AVAILABLE:
            return

        # Prefer the external `autonomous_car` artifacts when enabled.
        if 'USE_EXTERNAL_AUTONOMOUS_CAR_MODEL' in globals() and USE_EXTERNAL_AUTONOMOUS_CAR_MODEL:
            try:
                import joblib

                if not os.path.exists(EXTERNAL_MODEL_FILE):
                    print(f"[ERR] External model not found: {EXTERNAL_MODEL_FILE}")
                    return
                if not os.path.exists(EXTERNAL_SCALER_FILE):
                    print(f"[ERR] External scaler not found: {EXTERNAL_SCALER_FILE}")
                    return
                if not os.path.exists(EXTERNAL_LABELS_FILE):
                    print(f"[ERR] External labels not found: {EXTERNAL_LABELS_FILE}")
                    return

                self.ai_model = load_model(EXTERNAL_MODEL_FILE)
                self.ai_scaler = joblib.load(EXTERNAL_SCALER_FILE)
                self.ai_encoder = np.load(EXTERNAL_LABELS_FILE, allow_pickle=True)

                print(f"[OK] Loaded EXTERNAL AI model from {EXTERNAL_MODEL_FILE}")
                print(f"[OK] Loaded EXTERNAL scaler from {EXTERNAL_SCALER_FILE}")
                print(f"[OK] Loaded EXTERNAL label classes from {EXTERNAL_LABELS_FILE}")
                return
            except Exception as e:
                print(f"[ERR] Error loading external AI artifacts: {e}")
                self.ai_model = None
                self.ai_scaler = None
                self.ai_encoder = None
                # Fall back to local artifacts below.

        # Local (Driving project) artifacts fallback.
        if os.path.exists(MODEL_FILE):
            try:
                self.ai_model = load_model(MODEL_FILE)
                print(f"[OK] Loaded AI model from {MODEL_FILE}")

                with open(LABEL_ENCODER_FILE, "rb") as f:
                    self.ai_encoder = pickle.load(f)
                print("[OK] Loaded label encoder")

                with open(SCALER_FILE, "rb") as f:
                    self.ai_scaler = pickle.load(f)
                print("[OK] Loaded feature scaler")

            except Exception as e:
                print(f"[ERR] Error loading AI model: {e}")
                self.ai_model = None
        else:
            print("[INFO] AI model not found. Train model first using train_lstm_model.py")
    
    def handle_events(self):
        """Handle user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                
                # Mode switching
                elif event.key == pygame.K_t:
                    self.teacher_mode = True
                    self.ai_mode = False
                    print("[MODE] Switched to TEACHER mode")
                
                elif event.key == pygame.K_a:
                    if self.ai_model:
                        self.teacher_mode = False
                        self.ai_mode = True
                        print("[MODE] Switched to AI mode")
                    else:
                        print("[ERR] AI model not loaded!")
                
                elif event.key == pygame.K_m:
                    self.teacher_mode = False
                    self.ai_mode = False
                    print("[MODE] Switched to MANUAL mode")
                
                # Density control
                elif event.key == pygame.K_LEFTBRACKET:
                    self.density_manager.cycle_density_down()
                    self.traffic.update_density(
                        self.density_manager.get_spawn_parameters()['num_cars']
                    )
                    print(f"[DENSITY] Traffic density: {self.density_manager.current_density}")
                
                elif event.key == pygame.K_RIGHTBRACKET:
                    self.density_manager.cycle_density_up()
                    self.traffic.update_density(
                        self.density_manager.get_spawn_parameters()['num_cars']
                    )
                    print(f"[DENSITY] Traffic density: {self.density_manager.current_density}")
    
    def update(self):
        """Update simulation state"""
        self.frame_count += 1
        
        # Update road scroll
        road_scroll_speed = self.car.speed
        self.road_offset_1 += road_scroll_speed
        self.road_offset_2 += road_scroll_speed
        
        # Wrap road offsets
        if self.road_offset_1 >= HEIGHT:
            self.road_offset_1 = self.road_offset_2 - HEIGHT
        if self.road_offset_2 >= HEIGHT:
            self.road_offset_2 = self.road_offset_1 - HEIGHT
        if self.road_offset_1 <= -HEIGHT:
            self.road_offset_1 = self.road_offset_2 + HEIGHT
        if self.road_offset_2 <= -HEIGHT:
            self.road_offset_2 = self.road_offset_1 + HEIGHT
        
        # Update surrounding vehicles tracker
        self.surrounding_vehicles.update(
            self.traffic.cars,
            {'x': self.car.x + CAR_WIDTH // 2, 'y': self.car.y}
        )
        
        # Update safety analysis
        ego_state = self.car.vehicle_state.get_state_vector()
        self.safety_analyzer.analyze_all_actions(ego_state, self.surrounding_vehicles)
        
        # Update FSM
        self.fsm_controller.update(
            ego_state,
            self.surrounding_vehicles,
            self.safety_analyzer,
            self.frame_count
        )
        
        # Update car
        move_result = self.car.move(
            is_ai=self.ai_mode,
            model=self.ai_model,
            encoder=self.ai_encoder,
            scaler=self.ai_scaler,
            is_teacher=self.teacher_mode,
            fsm_controller=self.fsm_controller,
            safety_analyzer=self.safety_analyzer
        )
        
        if not move_result:
            print("\n[CRASH] COLLISION DETECTED!")
            self.display_crash_screen()
            self.running = False
            return
        
        # Keep car at fixed y position
        self.car.y = HEIGHT - self.car.car_height - 100
        
        # Update traffic
        new_overtakes = self.traffic.update(road_scroll_speed, self.car.y)
        self.total_overtakes += new_overtakes
        
        # Update distance
        if self.car.speed > 0:
            self.distance_traveled += self.car.speed * 0.0833
        
        # Collect training data (if not in AI mode)
        if not self.ai_mode:
            self._collect_training_data()
        
        # Log density periodically
        self.density_manager.update_periodic_log(
            time.time(),
            self.surrounding_vehicles.get_vehicle_count()
        )
    
    def _collect_training_data(self):
        """Collect training data for LSTM"""
        row = [r['offset'] for r in self.car.sensor.readings]
        
        if self.car.controls['left']:
            action = "left"
        elif self.car.controls['right']:
            action = "right"
        elif self.car.controls['reverse']:
            action = "slow_down"
        else:
            action = "forward"
        
        row.append(action)
        self.current_run_data.append(row)
    
    def render(self):
        """Render all graphics"""
        # Clear screen
        self.screen.fill(WHITE)
        
        # Draw road
        self.road.draw(self.screen, self.road_offset_1, self.road_offset_2)
        
        # Draw traffic
        self.traffic.draw(self.screen)
        
        # Draw car
        self.car.draw(self.screen)
        
        # Draw HUD
        self._draw_hud()
        
        # Update display
        pygame.display.flip()
    
    def _draw_hud(self):
        """Draw all HUD elements"""
        # Mode info
        if self.teacher_mode:
            mode_name = "TEACHER (REC)"
        elif self.ai_mode:
            mode_name = "AI AUTO"
        else:
            mode_name = "MANUAL (REC)"
        
        mode_info = {
            'mode_name': mode_name,
            'is_teacher': self.teacher_mode,
            'is_ai': self.ai_mode
        }
        
        # Stats
# Stats
        stats = {
            'frames': len(self.current_run_data),
            'distance': self.distance_traveled,
            'overtakes': self.total_overtakes,
            'speed': self.car.speed * 10,
            'lane': self.car.vehicle_state.lane,
            'auto_run_time_remaining': None  # Add this
        }
        
        # Add auto-run timer
        if self.auto_run_enabled and self.auto_run_start_time:
            elapsed = time.time() - self.auto_run_start_time
            stats['auto_run_time_remaining'] = AUTO_RUN_TOTAL_TIME - elapsed  
        
        # Draw panels
        self.hud.draw_main_panel(self.screen, mode_info, stats)
        self.hud.draw_vehicle_state_panel(self.screen, self.car.vehicle_state)
        self.hud.draw_safety_panel(self.screen, self.safety_analyzer)
        self.hud.draw_fsm_state(self.screen, self.fsm_controller)
        self.hud.draw_surrounding_vehicles_panel(self.screen, self.surrounding_vehicles)
        self.hud.draw_density_control(self.screen, self.density_manager)
    
    def display_crash_screen(self):
        """Display crash message"""
        font = pygame.font.SysFont("Impact", 80)
        message = font.render("CRASHED!", True, RED)
        self.screen.blit(
            message,
            (WIDTH // 2 - message.get_width() // 2, HEIGHT // 2 - message.get_height() // 2)
        )
        pygame.display.flip()
        pygame.time.wait(2000)
    
    def save_data(self):
        """Save training data and logs"""
        # Save training data
        if len(self.current_run_data) > 0:
            file_exists = os.path.isfile(TRAINING_DATA_FILE)
            
            try:
                with open(TRAINING_DATA_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    
                    if not file_exists:
                        num_sensors = len(self.current_run_data[0]) - 1
                        headers = [f"sensor_{i+1}" for i in range(num_sensors)] + ["decision"]
                        writer.writerow(headers)
                    
                    writer.writerows(self.current_run_data)
                
                print(f"\n[OK] Saved {len(self.current_run_data)} training samples to {TRAINING_DATA_FILE}")
            
            except PermissionError:
                print(f"\n[ERR] Could not save {TRAINING_DATA_FILE}. File may be open.")
        
        # Save density log
        self.density_manager.save_density_log()
    def _handle_auto_run(self):
        """Handle automatic density changes and termination"""
        elapsed = time.time() - self.auto_run_start_time
        
        # Check if total time exceeded
        if elapsed >= AUTO_RUN_TOTAL_TIME:
            print(f"\n[OK] AUTO-RUN COMPLETE: {AUTO_RUN_TOTAL_TIME}s elapsed")
            self.running = False
            return
        
        # Find current schedule
        cumulative_time = 0
        for i, schedule in enumerate(AUTO_RUN_DENSITY_SCHEDULE):
            cumulative_time += schedule['duration']
            
            if elapsed < cumulative_time:
                # Check if we need to change density
                if i != self.auto_run_schedule_index:
                    self.auto_run_schedule_index = i
                    target_density = schedule['density']
                    
                    if self.density_manager.current_density != target_density:
                        self.density_manager.set_density(target_density)
                        self.traffic.update_density(
                            self.density_manager.get_spawn_parameters()['num_cars']
                        )
                        print(f"\n⏱️ AUTO-RUN: Switched to {target_density} density at {elapsed:.1f}s")
                break
    def run(self):
        """Main simulation loop"""
        
        # Start auto-run timer
        if self.auto_run_enabled:
            self.auto_run_start_time = time.time()
        
        while self.running:
            self.handle_events()
            
            # Auto-run logic
            if self.auto_run_enabled:
                self._handle_auto_run()
            
            self.update()
            self.render()
            self.clock.tick(FPS)
        
        # Cleanup
        pygame.quit()
        self.save_data()
        
        print("\n" + "="*60)
        print("SIMULATION ENDED")
        print("="*60)
        print(f"Total frames: {self.frame_count}")
        print(f"Distance traveled: {self.distance_traveled:.0f} m")
        print(f"Overtakes: {self.total_overtakes}")
        print("="*60 + "\n")


def main():
    """Entry point"""
    print("\n")
    # Use ASCII-only banner for Windows consoles that default to cp1252.
    print("+" + "="*58 + "+")
    print("|" + " "*58 + "|")
    print("|" + "  ADVANCED AUTONOMOUS DRIVING SIMULATION".center(58) + "|")
    print("|" + "  LSTM + FSM + Safety Analysis + Density Control".center(58) + "|")
    print("|" + " "*58 + "|")
    print("+" + "="*58 + "+")
    
    sim = AdvancedSimulation()
    sim.run()


if __name__ == "__main__":
    main()