import pygame

class Lander:
    def __init__(self, screen_width, screen_height):
        self.sw = screen_width
        self.sh = screen_height
        self.x = screen_width // 2
        self.y = 50
        self.vx = 0.0
        self.angle = 0.0
        
    def update(self, tilt_angle, depth_Z):
        """
        Derives lander state explicitly from CV telemetry.
        """
        # 1. Update Rotation strictly from CV
        self.angle = tilt_angle
        
        # 2. Update Altitude strictly from CV
        surface_y = self.sh - 100
        ceiling_y = 50
        max_z = 250.0
        
        # Linear map: Z=max_z -> ceiling_y, Z=0 -> surface_y
        mapped_y = surface_y - (depth_Z / max_z) * (surface_y - ceiling_y)
        self.y = max(ceiling_y, min(surface_y, mapped_y))
        
        # 3. Update Lateral from Input
        self.x += self.vx
        self.x = max(50, min(self.sw - 50, self.x))
        
    def draw(self, screen, thrusting=False):
        size = 20
        points = [
            pygame.math.Vector2(0, -size * 1.5),
            pygame.math.Vector2(-size, size),
            pygame.math.Vector2(size, size)
        ]
        
        # Pygame rotates clockwise, math tilt is counter-clockwise
        rotated_points = [p.rotate(self.angle) for p in points]
        screen_points = [(self.x + p.x, self.y + p.y) for p in rotated_points]
        
        pygame.draw.polygon(screen, (200, 200, 200), screen_points)
        pygame.draw.polygon(screen, (255, 255, 255), screen_points, 2)
        
        if thrusting:
            flame = [
                pygame.math.Vector2(-size*0.5, size),
                pygame.math.Vector2(size*0.5, size),
                pygame.math.Vector2(0, size*1.5 + (pygame.time.get_ticks() % 10))
            ]
            r_flame = [p.rotate(self.angle) for p in flame]
            s_flame = [(self.x + p.x, self.y + p.y) for p in r_flame]
            pygame.draw.polygon(screen, (255, 100, 0), s_flame)
