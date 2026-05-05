import pygame
import sys
import numpy as np
import random
from lander import Lander
from hud_pygame import HUD
from integration import CVSystem

def draw_background(screen):
    screen.fill((5, 5, 15)) # Dark space
    
    # Stars
    random.seed(42) # fixed stars
    for _ in range(100):
        x = random.randint(0, screen.get_width())
        y = random.randint(0, screen.get_height() - 100)
        pygame.draw.circle(screen, (255, 255, 255), (x, y), random.randint(1, 2))
        
    # Lunar surface
    points = [(0, screen.get_height()), (0, screen.get_height() - 80)]
    for x in range(0, screen.get_width(), 50):
        points.append((x, screen.get_height() - 80 + random.randint(-15, 15)))
    points.append((screen.get_width(), screen.get_height() - 80))
    points.append((screen.get_width(), screen.get_height()))
    
    pygame.draw.polygon(screen, (100, 100, 100), points)
    
    # Landing Pad (Target)
    pad_center = screen.get_width() // 2
    pad_y = screen.get_height() - 90
    pygame.draw.rect(screen, (0, 255, 0), (pad_center - 40, pad_y, 80, 10))

def main():
    pygame.init()
    sw, sh = 1024, 768
    screen = pygame.display.set_mode((sw, sh))
    pygame.display.set_caption("Lunar Landing Simulator (CV Powered)")
    clock = pygame.time.Clock()
    
    cv_system = CVSystem()
    lander = Lander(sw, sh)
    hud = HUD()
    
    # Simulation targets for CV
    sim_z = 2.5 # start at 2.5 meters
    sim_sun_az = 0.0
    sim_sun_el = np.radians(45)
    sim_panel_p = 0.0
    sim_panel_r = 0.0
    
    game_state = {
        'landed': False,
        'crashed': False
    }
    
    running = True
    while running:
        thrusting = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        keys = pygame.key.get_pressed()
        
        if not (game_state['landed'] or game_state['crashed']):
            if keys[pygame.K_w]:
                sim_z += 0.05 # thrust up (increase altitude)
                thrusting = True
            if keys[pygame.K_a]:
                lander.vx = -3.0
            elif keys[pygame.K_d]:
                lander.vx = 3.0
            else:
                lander.vx = 0.0
                
            if keys[pygame.K_SPACE]:
                # Auto stabilize -> snap panel to sun vector to reduce error
                # In game terms, we want to pitch panel towards 0 to stabilize
                sim_panel_p *= 0.8
                sim_panel_r *= 0.8
            else:
                # Add drift to tilt if no stabilization
                sim_panel_p += np.random.normal(0, 0.2)
                sim_panel_r += np.random.normal(0, 0.2)
                
            # Gravity
            if not thrusting:
                sim_z -= 0.02
        
        if keys[pygame.K_r]:
            # Reset
            sim_z = 2.5
            sim_panel_p = 0.0
            sim_panel_r = 0.0
            lander.x = sw // 2
            game_state['landed'] = False
            game_state['crashed'] = False
            
        # 1. RUN CV PIPELINE
        cv_data = cv_system.update_from_cv(sim_z, sim_sun_az, sim_sun_el, sim_panel_p, sim_panel_r)
        
        # 2. BIND CV OUTPUTS TO PHYSICS
        # Lander's visuals and altitude mapping are strictly driven by the CV depth & tilt!
        lander.update(cv_data['tilt_angle'], cv_data['depth_Z'])
        
        # 3. WIN / FAIL LOGIC
        if not game_state['landed'] and not game_state['crashed']:
            # Check Crash
            if abs(cv_data['tilt_angle']) > 30.0 or cv_data['confidence'] == 'LOW':
                if cv_data['depth_Z'] < 100: # only crash if somewhat low
                    game_state['crashed'] = True
            
            # Check Success
            lateral_error = abs(lander.x - sw//2)
            if cv_data['depth_Z'] < 10.0:
                if abs(cv_data['tilt_angle']) < 10.0 and lateral_error < 50 and cv_data['confidence'] == 'HIGH' and cv_data['alignment_error'] < 15.0:
                    game_state['landed'] = True
                else:
                    game_state['crashed'] = True
                    
        # 4. RENDER
        draw_background(screen)
        lander.draw(screen, thrusting)
        hud.draw(screen, cv_data, game_state)
        
        pygame.display.flip()
        clock.tick(30) # Will likely run slower due to CV pipeline
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
