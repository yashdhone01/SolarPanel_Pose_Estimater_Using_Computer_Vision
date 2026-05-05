import pygame

class HUD:
    def __init__(self, font_name='Courier', size=24):
        pygame.font.init()
        self.font = pygame.font.SysFont(font_name, size, bold=True)
        self.large_font = pygame.font.SysFont(font_name, 64, bold=True)
        
    def draw(self, screen, cv_data, game_state):
        y = 20
        def draw_text(text, color, pos):
            surface = self.font.render(text, True, color)
            screen.blit(surface, pos)
            return pos[1] + 30
            
        # Draw Telemetry
        draw_text("CV TELEMETRY", (255, 255, 0), (20, y))
        y += 30
        y = draw_text(f"Altitude:      {cv_data['depth_Z']:.1f} cm", (255, 255, 255), (20, y))
        y = draw_text(f"Tilt Angle:    {cv_data['tilt_angle']:.1f} deg", (255, 255, 255), (20, y))
        y = draw_text(f"Align Error:   {cv_data['alignment_error']:.1f} deg", (255, 255, 255), (20, y))
        
        conf_col = (0, 255, 0) if cv_data['confidence'] == 'HIGH' else (255, 0, 0)
        y = draw_text(f"Confidence:    {cv_data['confidence']}", conf_col, (20, y))
        
        haz_col = (255, 0, 0) if cv_data['hazard'] else (0, 255, 0)
        y = draw_text(f"Hazard Detect: {cv_data['hazard']}", haz_col, (20, y))
        
        # Guidance
        guidance = "GOOD"
        guidance_col = (0, 255, 0)
        if cv_data['alignment_error'] > 5.0:
            guidance = "ADJUST ALIGNMENT"
            guidance_col = (255, 255, 0)
        elif cv_data['depth_Z'] > 50:
            guidance = "DESCEND"
            
        y += 20
        draw_text(f"GUIDANCE: {guidance}", guidance_col, (20, y))
        
        # Controls
        y = screen.get_height() - 150
        draw_text("CONTROLS", (255, 255, 0), (20, y))
        y += 30
        y = draw_text("[W]   Thrust Up", (255, 255, 255), (20, y))
        y = draw_text("[A/D] Lateral Drift", (255, 255, 255), (20, y))
        y = draw_text("[SPC] Auto-Stabilize", (255, 255, 255), (20, y))
        y = draw_text("[R]   Reset", (255, 255, 255), (20, y))
        
        # Game State Overlays
        if game_state['crashed']:
            s = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
            s.fill((255, 0, 0, 100))
            screen.blit(s, (0,0))
            text = self.large_font.render("CRASH ⚠", True, (255, 255, 255))
            screen.blit(text, (screen.get_width()//2 - text.get_width()//2, screen.get_height()//2))
            
        elif game_state['landed']:
            s = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
            s.fill((0, 255, 0, 100))
            screen.blit(s, (0,0))
            text = self.large_font.render("SUCCESSFUL LANDING 🚀", True, (255, 255, 255))
            screen.blit(text, (screen.get_width()//2 - text.get_width()//2, screen.get_height()//2))
