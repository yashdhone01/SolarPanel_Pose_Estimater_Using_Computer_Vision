import pygame
import sys
import math

# --- CV Integration ---
cv_data = {
    'depth_Z': None,
    'tilt_angle': None,
    'alignment_error': None,
    'confidence': None
}

def update_from_cv(depth_Z, tilt_angle, alignment_error, confidence):
    global cv_data
    cv_data['depth_Z'] = depth_Z
    cv_data['tilt_angle'] = tilt_angle
    cv_data['alignment_error'] = alignment_error
    cv_data['confidence'] = confidence

# --- Game Logic ---
def main():
    pygame.init()
    sw, sh = 800, 600
    screen = pygame.display.set_mode((sw, sh))
    pygame.display.set_caption("Lunar Lander Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 20, bold=True)
    large_font = pygame.font.SysFont('Arial', 48, bold=True)
    
    # Physics constants
    GRAVITY = 0.05
    THRUST = 0.15
    ROT_SPEED = 3.0
    LATERAL_THRUST = 0.1
    MAX_SPEED = 10.0
    
    def reset_lander():
        return {
            'x': sw / 2,
            'y': 50.0,
            'vx': 0.0,
            'vy': 0.0,
            'angle': 0.0,
            'state': 'FLYING', # FLYING, SUCCESS, CRASH
            'timer': 0
        }
        
    lander = reset_lander()
    pad_x, pad_w = sw / 2, 100
    ground_y = sh - 50
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        keys = pygame.key.get_pressed()
        
        # Apply CV overrides if available
        if cv_data['depth_Z'] is not None:
            # Map depth 0-250cm to screen Y
            mapped_y = ground_y - (cv_data['depth_Z'] / 250.0) * (ground_y - 50)
            lander['y'] = max(50.0, min(float(ground_y), mapped_y))
            lander['vy'] = 0 
            
        if cv_data['tilt_angle'] is not None:
            lander['angle'] = cv_data['tilt_angle']
            
        if lander['state'] == 'FLYING':
            # Controls
            if keys[pygame.K_w]:
                lander['vy'] -= THRUST
            if keys[pygame.K_a]:
                lander['vx'] -= LATERAL_THRUST
            if keys[pygame.K_d]:
                lander['vx'] += LATERAL_THRUST
            if keys[pygame.K_LEFT]:
                lander['angle'] -= ROT_SPEED
            if keys[pygame.K_RIGHT]:
                lander['angle'] += ROT_SPEED
                
            # Physics
            if cv_data['depth_Z'] is None:
                lander['vy'] += GRAVITY
                
            lander['vx'] = max(-MAX_SPEED, min(MAX_SPEED, lander['vx']))
            lander['vy'] = max(-MAX_SPEED, min(MAX_SPEED, lander['vy']))
            
            lander['x'] += lander['vx']
            if cv_data['depth_Z'] is None:
                lander['y'] += lander['vy']
            
            # Bounds
            lander['x'] = max(20.0, min(sw - 20.0, lander['x']))
            if lander['y'] <= 20:
                lander['y'] = 20.0
                lander['vy'] = 0
                
            # Landing Check
            if lander['y'] >= ground_y - 15: # 15 is lander half-height
                lander['y'] = ground_y - 15
                
                speed = math.hypot(lander['vx'], lander['vy'])
                in_pad = (pad_x - pad_w/2) <= lander['x'] <= (pad_x + pad_w/2)
                tilt_ok = abs(lander['angle']) < 10.0
                speed_ok = speed < 3.0
                
                if in_pad and tilt_ok and speed_ok:
                    lander['state'] = 'SUCCESS'
                else:
                    lander['state'] = 'CRASH'
                    
                lander['timer'] = pygame.time.get_ticks()
                
        else:
            # Wait 2 seconds and reset
            if pygame.time.get_ticks() - lander['timer'] > 2000:
                lander = reset_lander()
                
        # Draw Background
        screen.fill((10, 10, 20)) # Dark space
        pygame.draw.rect(screen, (50, 50, 50), (0, ground_y, sw, sh - ground_y))
        pygame.draw.rect(screen, (0, 200, 0), (pad_x - pad_w/2, ground_y, pad_w, 10))
        
        # Draw Lander
        points = [pygame.math.Vector2(0, -20), pygame.math.Vector2(-15, 15), pygame.math.Vector2(15, 15)]
        rotated = [p.rotate(lander['angle']) for p in points]
        screen_points = [(lander['x'] + p.x, lander['y'] + p.y) for p in rotated]
        pygame.draw.polygon(screen, (200, 200, 200), screen_points)
        
        # HUD
        alt = max(0, ground_y - 15 - lander['y'])
        speed = math.hypot(lander['vx'], lander['vy'])
        
        y_off = 20
        screen.blit(font.render(f"Altitude: {alt:.1f}", True, (255,255,255)), (20, y_off)); y_off+=25
        screen.blit(font.render(f"Speed: {speed:.1f}", True, (255,255,255)), (20, y_off)); y_off+=25
        screen.blit(font.render(f"Tilt: {lander['angle']:.1f}", True, (255,255,255)), (20, y_off)); y_off+=25
        screen.blit(font.render(f"Status: {lander['state']}", True, (255,255,0)), (20, y_off))
        
        if lander['state'] == 'SUCCESS':
            txt = large_font.render("LANDED SUCCESS", True, (0, 255, 0))
            screen.blit(txt, (sw/2 - txt.get_width()/2, sh/2 - txt.get_height()/2))
        elif lander['state'] == 'CRASH':
            txt = large_font.render("CRASH", True, (255, 0, 0))
            screen.blit(txt, (sw/2 - txt.get_width()/2, sh/2 - txt.get_height()/2))
            
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
