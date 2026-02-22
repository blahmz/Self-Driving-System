"""
Traffic Density Manager
Handles dynamic traffic density control and logging
"""

import csv
import os
import time
from config import DENSITY_LEVELS, DEFAULT_DENSITY, DENSITY_LOG_FILE


class DensityManager:
    """Manages traffic density levels and logging"""
    
    def __init__(self):
        self.current_density = DEFAULT_DENSITY
        self.density_config = DENSITY_LEVELS[self.current_density].copy()
        self.density_log = []
        self.last_log_time = time.time()
        
    def set_density(self, density_level):
        """Change traffic density level"""
        if density_level in DENSITY_LEVELS:
            self.current_density = density_level
            self.density_config = DENSITY_LEVELS[density_level].copy()
            
            # Log the change
            self.log_density_change(time.time())
            
            return True
        return False
    
    def cycle_density_up(self):
        """Cycle to next higher density level"""
        levels = ['LOW', 'MEDIUM', 'HIGH']
        current_idx = levels.index(self.current_density)
        next_idx = (current_idx + 1) % len(levels)
        self.set_density(levels[next_idx])
    
    def cycle_density_down(self):
        """Cycle to next lower density level"""
        levels = ['LOW', 'MEDIUM', 'HIGH']
        current_idx = levels.index(self.current_density)
        next_idx = (current_idx - 1) % len(levels)
        self.set_density(levels[next_idx])
    
    def get_spawn_parameters(self):
        """Get current spawn parameters"""
        return self.density_config
    
    def log_density_change(self, timestamp):
        """Log density change with timestamp"""
        log_entry = {
            'timestamp': timestamp,
            'density': self.current_density,
            'num_cars': self.density_config['num_cars'],
            'spawn_prob': self.density_config['spawn_prob'],
            'min_gap': self.density_config['min_gap']
        }
        self.density_log.append(log_entry)
    
    def update_periodic_log(self, current_time, vehicle_count):
        """Log density state periodically (every second)"""
        if current_time - self.last_log_time >= 1.0:
            log_entry = {
                'timestamp': current_time,
                'density': self.current_density,
                'actual_vehicle_count': vehicle_count,
                'target_vehicle_count': self.density_config['num_cars']
            }
            self.density_log.append(log_entry)
            self.last_log_time = current_time
    
    def save_density_log(self):
        """Save density log to CSV file"""
        if not self.density_log:
            return
        
        try:
            file_exists = os.path.isfile(DENSITY_LOG_FILE)
            
            with open(DENSITY_LOG_FILE, mode='a', newline='') as f:
                fieldnames = ['timestamp', 'density', 'num_cars', 'spawn_prob', 
                            'min_gap', 'actual_vehicle_count', 'target_vehicle_count']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                for entry in self.density_log:
                    # Fill missing fields with None
                    row = {field: entry.get(field, None) for field in fieldnames}
                    writer.writerow(row)
            
            print(f"Density log saved: {len(self.density_log)} entries")
            self.density_log.clear()
            
        except Exception as e:
            print(f"Error saving density log: {e}")
    
    def get_density_info_text(self):
        """Get formatted density info for display"""
        return f"{self.current_density} ({self.density_config['num_cars']} cars)"