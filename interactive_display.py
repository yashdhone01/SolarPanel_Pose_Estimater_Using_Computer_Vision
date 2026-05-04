import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

class InteractiveDisplay:
    def __init__(self):
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 9), facecolor='#111111')
        self.axes = {}
        
        self.axes['panel1'] = self.fig.add_subplot(231)
        self.axes['panel2'] = self.fig.add_subplot(232)
        self.axes['panel3'] = self.fig.add_subplot(233)
        self.axes['panel4'] = self.fig.add_subplot(234)
        self.axes['panel5'] = self.fig.add_subplot(235)
        self.axes['panel6'] = self.fig.add_subplot(236, projection='3d')
        self.axes['panel6'].set_facecolor('#111111')
        
        self.fig.tight_layout(pad=3.0)
        plt.ion() # Interactive mode on
        
        self.last_key = None
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Keep track of latest normal and angle to avoid flickering or wiping panel 6 when only sun updates
        self.last_normal = np.array([0., 0., 1.])
        
    def _on_key_press(self, event):
        self.last_key = event.key
        
    def get_key(self):
        k = self.last_key
        self.last_key = None
        return k

    def init_display(self):
        plt.show(block=False)

    def update_sun_only(self, sun_vector):
        """
        Updates ONLY the 3D Panel 6.
        """
        self._draw_panel6(self.last_normal, sun_vector)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update_full_display(self, data_dict):
        """
        Updates all 6 subplots with the provided data dictionary.
        """
        left_img = data_dict.get('left_img')
        right_img = data_dict.get('right_img')
        matches_img = data_dict.get('matches_img')
        disparity_map = data_dict.get('disparity_map')
        est_pitch_roll = data_dict.get('estimated_pitch_roll', (0.0, 0.0))
        gt_pitch_roll = data_dict.get('ground_truth_pitch_roll', (0.0, 0.0))
        normal_vector = data_dict.get('normal_vector')
        sun_vector = data_dict.get('sun_vector')
        
        if normal_vector is not None:
             self.last_normal = normal_vector
        else:
             normal_vector = self.last_normal
             
        # 1. Left Image
        self.axes['panel1'].clear()
        if left_img is not None:
            self.axes['panel1'].imshow(left_img, cmap='gray')
        self.axes['panel1'].set_title("Left view")
        self.axes['panel1'].axis('off')
        
        # 2. Right Image
        self.axes['panel2'].clear()
        if right_img is not None:
            self.axes['panel2'].imshow(right_img, cmap='gray')
        self.axes['panel2'].set_title("Right view")
        self.axes['panel2'].axis('off')
        
        # 3. SIFT Matches
        self.axes['panel3'].clear()
        if matches_img is not None:
            self.axes['panel3'].imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
        self.axes['panel3'].set_title(f"SIFT Matches")
        self.axes['panel3'].axis('off')
        
        # 4. Disparity Map
        self.axes['panel4'].clear()
        if disparity_map is not None:
            self.axes['panel4'].imshow(disparity_map, cmap='plasma')
        self.axes['panel4'].set_title("Disparity map")
        self.axes['panel4'].axis('off')
        
        # 5. Orientation comparison
        self.axes['panel5'].clear()
        labels = ['Pitch', 'Roll']
        gt_vals = [gt_pitch_roll[0], gt_pitch_roll[1]]
        est_vals = [est_pitch_roll[0], est_pitch_roll[1]]
        x = np.arange(len(labels))
        width = 0.35
        self.axes['panel5'].bar(x - width/2, gt_vals, width, label='GT', color='green')
        self.axes['panel5'].bar(x + width/2, est_vals, width, label='Est', color='red')
        self.axes['panel5'].set_ylabel('Degrees')
        self.axes['panel5'].set_title("Estimated vs GT")
        self.axes['panel5'].set_xticks(x)
        self.axes['panel5'].set_xticklabels(labels)
        self.axes['panel5'].legend()
        self.axes['panel5'].set_ylim([-60, 60])
        self.axes['panel5'].axhline(0, color='black', linewidth=0.5)
        
        # 6. Sun + Panel visualization
        self._draw_panel6(normal_vector, sun_vector)

        # Force draw (don't block)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _draw_panel6(self, normal_vector, sun_vector):
        self.axes['panel6'].clear()
        if normal_vector is not None and sun_vector is not None:
            # Normalise vectors just in case
            n = normal_vector / np.linalg.norm(normal_vector)
            s = sun_vector / np.linalg.norm(sun_vector)
            
            # Compute angle
            dot_prod = np.dot(n, s)
            angle_rad = np.arccos(np.clip(dot_prod, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)
            
            ax3d = self.axes['panel6']
            
            # Draw panel plane (generic 1x1 plane at origin oriented to normal)
            if abs(n[0]) > 0.1 or abs(n[1]) > 0.1:
                u = np.array([-n[1], n[0], 0])
            else:
                u = np.array([1, 0, 0])
            u = u / np.linalg.norm(u)
            v = np.cross(n, u)
            
            grid_size = 0.5
            uu, vv = np.meshgrid([-grid_size, grid_size], [-grid_size, grid_size])
            
            X = uu * u[0] + vv * v[0]
            Y = uu * u[1] + vv * v[1]
            Z = uu * u[2] + vv * v[2]
            
            ax3d.plot_surface(X, Y, Z, alpha=0.9, color='#1f77b4', edgecolor='silver', linewidth=1.5)
            
            # Draw Normal Vector
            ax3d.quiver(0, 0, 0, n[0], n[1], n[2], length=1.0, normalize=True, color='blue', label='Normal', arrow_length_ratio=0.1)
            
            # Draw Sun Vector
            ax3d.quiver(0, 0, 0, s[0], s[1], s[2], length=1.0, normalize=True, color='orange', label='Sun', arrow_length_ratio=0.1)
            
            # Draw Sun Point
            ax3d.scatter([s[0]*1.2], [s[1]*1.2], [s[2]*1.2], color='orange', s=100, marker='o')
            
            ax3d.set_title(f"Sun & Panel Alignment\nAlignment Error: {angle_deg:.1f}°")
            ax3d.set_xlabel('X')
            ax3d.set_ylabel('Y')
            ax3d.set_zlabel('Z')
            
            ax3d.set_xlim([-1, 1])
            ax3d.set_ylim([-1, 1])
            ax3d.set_zlim([-1, 1])
            ax3d.legend()


# Global instance for easy use
_display_instance = None

def init_display():
    global _display_instance
    if _display_instance is None:
        _display_instance = InteractiveDisplay()
        _display_instance.init_display()

def update_full_display(data_dict):
    if _display_instance is not None:
        _display_instance.update_full_display(data_dict)
        
def update_sun_only(sun_vector):
    if _display_instance is not None:
        _display_instance.update_sun_only(sun_vector)
        
def get_key():
    if _display_instance is not None:
        return _display_instance.get_key()
    return None
