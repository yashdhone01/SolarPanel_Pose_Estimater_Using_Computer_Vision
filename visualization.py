import matplotlib
matplotlib.use('Agg') # Off-screen rendering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

class SceneRenderer3D:
    def __init__(self):
        plt.style.use('dark_background')
        # Setting DPI and figsize to approximate 640x480 natively for speed
        self.fig = plt.figure(figsize=(6.4, 4.8), dpi=100, facecolor='#111111')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#111111')
        
    def render(self, normal_vector, sun_vector, target_size=(640, 480)):
        self.ax.clear()
        
        n = normal_vector / np.linalg.norm(normal_vector)
        s = sun_vector / np.linalg.norm(sun_vector)
        
        # Draw panel plane
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
        
        self.ax.plot_surface(X, Y, Z, alpha=0.9, color='#1f77b4', edgecolor='silver', linewidth=1.5)
        
        # Draw Vectors
        self.ax.quiver(0, 0, 0, n[0], n[1], n[2], length=1.0, normalize=True, color='#4da6ff', label='Normal', arrow_length_ratio=0.1)
        self.ax.quiver(0, 0, 0, s[0], s[1], s[2], length=1.0, normalize=True, color='#ff9933', label='Sun', arrow_length_ratio=0.1)
        self.ax.scatter([s[0]*1.2], [s[1]*1.2], [s[2]*1.2], color='#ff9933', s=100, marker='o')
        
        self.ax.set_title("3D Reconstruction from Stereo Geometry", color='white', pad=10)
        
        # Add watermark text natively using ax.text2D
        self.ax.text2D(0.05, 0.95, "Derived from real-time stereo reconstruction", transform=self.ax.transAxes, color='gray', fontsize=8)
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])
        
        # Render to array
        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        
        # Matplotlib is RGBA, OpenCV is BGR
        img_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        
        if (img_bgr.shape[1], img_bgr.shape[0]) != target_size:
            img_bgr = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
            
        return img_bgr

_renderer = None
def get_3d_render(normal_vector, sun_vector, target_size=(640, 480)):
    global _renderer
    if _renderer is None:
        _renderer = SceneRenderer3D()
    return _renderer.render(normal_vector, sun_vector, target_size)
