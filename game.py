import numpy as np

class LanderState:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.state = 'WAITING' # WAITING, DESCENDING, SUCCESS, CRASH
        self.true_altitude_cm = 250.0 # Start at 2.5 meters
        self.v_z = 0.0 # cm per frame
        self.target_x = 320
        self.target_y = 240
        self.lander_x = 320
        self.lander_y = 240
        self.success_frames = 0
        self.frozen = False
        
        # Metrics computed
        self.error_cm = 0.0
        self.guidance_msg = "AWAITING TARGET"
        self.guidance_color = (255, 255, 255)
        
    def update(self, tilt, align_error, confidence, hazard, measured_depth_cm):
        if self.state in ['SUCCESS', 'CRASH']:
            return
            
        if self.state == 'DESCENDING':
            self.true_altitude_cm += self.v_z
            if self.true_altitude_cm < 0:
                self.true_altitude_cm = 0.0
                
        # Calculate XY error (using measured depth for validation!)
        dx = self.target_x - self.lander_x
        dy = self.target_y - self.lander_y
        
        # Pixels to cm using depth (assuming ~800px focal length)
        f_px = 800.0
        # If measured depth is invalid, fallback to true altitude for the sake of the game loop
        z = measured_depth_cm if measured_depth_cm > 1.0 else self.true_altitude_cm
        
        self.error_cm = np.sqrt(dx**2 + dy**2) * (z / f_px)
        
        if self.state == 'DESCENDING':
            # Guidance Decision Layer
            if tilt > 10.0:
                self.guidance_msg = "ADJUST TILT"
                self.guidance_color = (0, 255, 255) # Yellow
            elif self.error_cm > 10.0:
                self.guidance_msg = "MOVE LEFT/RIGHT"
                self.guidance_color = (0, 255, 255) # Yellow
            elif confidence == 'LOW' or hazard:
                self.guidance_msg = "HOLD - UNSTABLE"
                self.guidance_color = (0, 0, 255) # Red
            else:
                self.guidance_msg = "DESCEND"
                self.guidance_color = (0, 255, 0) # Green
                
            # Fail Conditions
            if hazard:
                self.state = 'CRASH'
                self.frozen = True
            elif self.true_altitude_cm < 10.0:
                # Win Conditions
                if self.error_cm < 2.0 and tilt < 5.0 and confidence == 'HIGH':
                    self.success_frames += 1
                    if self.success_frames > 2:
                        self.state = 'SUCCESS'
                        self.frozen = True
                else:
                    self.state = 'CRASH'
                    self.frozen = True
                    
    def input_thrust(self, delta):
        if not self.frozen:
            self.v_z += delta
            # Limit descent speed
            self.v_z = np.clip(self.v_z, -5.0, 5.0)
            
    def input_lateral(self, dx, dy):
        if not self.frozen:
            self.lander_x += dx
            self.lander_y += dy
            
    def set_target(self, x, y):
        if self.state == 'WAITING':
            self.target_x = x
            self.target_y = y
            self.state = 'DESCENDING'
            self.v_z = -1.0 # Auto start slow descent
