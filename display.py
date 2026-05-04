import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D projection
import numpy as np
import cv2

class Dashboard:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(15, 8))
        
        self.ax1 = self.fig.add_subplot(231)
        self.ax2 = self.fig.add_subplot(232)
        self.ax3 = self.fig.add_subplot(233)
        self.ax4 = self.fig.add_subplot(234)
        self.ax5 = self.fig.add_subplot(235)
        self.ax6 = self.fig.add_subplot(236, projection='3d')
        
        self.fig.tight_layout(pad=3.0)
        
    def update(self, img_l, img_r, match_img, disp_map, point_cloud, normal, centroid, err_pitch, err_roll, gt_pitch, gt_roll, est_pitch, est_roll, inlier_count):
        
        self.ax1.clear()
        self.ax1.imshow(img_l, cmap='gray')
        self.ax1.set_title("Left view")
        self.ax1.axis('off')
        
        self.ax2.clear()
        self.ax2.imshow(img_r, cmap='gray')
        self.ax2.set_title("Right view")
        self.ax2.axis('off')
        
        self.ax3.clear()
        if match_img is not None:
            # OpenCV BGR to RGB for matplotlib
            self.ax3.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
        self.ax3.set_title(f"SIFT matches — {inlier_count} inliers")
        self.ax3.axis('off')
        
        self.ax4.clear()
        if disp_map is not None:
            self.ax4.imshow(disp_map, cmap='plasma')
        self.ax4.set_title("Disparity map")
        self.ax4.axis('off')
        
        self.ax5.clear()
        labels = ['Pitch', 'Roll']
        gt_vals = [gt_pitch, gt_roll]
        est_vals = [est_pitch, est_roll]
        
        x = np.arange(len(labels))
        width = 0.35
        
        self.ax5.bar(x - width/2, gt_vals, width, label='GT', color='green')
        self.ax5.bar(x + width/2, est_vals, width, label='Est', color='red')
        
        self.ax5.set_ylabel('Degrees')
        self.ax5.set_title(f"Estimated vs GT")
        self.ax5.set_xticks(x)
        self.ax5.set_xticklabels(labels)
        self.ax5.legend()
        self.ax5.set_ylim([-60, 60])
        self.ax5.axhline(0, color='black', linewidth=0.5)
        
        self.ax6.clear()
        if point_cloud is not None and len(point_cloud) > 0:
            # Subsample point cloud for performance
            step = max(1, len(point_cloud) // 1000)
            subs = point_cloud[::step]
            self.ax6.scatter(subs[:, 0], subs[:, 1], subs[:, 2], c=subs[:, 2], cmap='viridis', s=2)
            
            # Plot the fitted plane
            if normal is not None and centroid is not None:
                # Define a grid around the centroid
                xx, yy = np.meshgrid(np.linspace(centroid[0]-0.2, centroid[0]+0.2, 10),
                                     np.linspace(centroid[1]-0.2, centroid[1]+0.2, 10))
                # Plane eq: A(x-cx) + B(y-cy) + C(z-cz) = 0
                # z = cz - (A(x-cx) + B(y-cy)) / C
                if abs(normal[2]) > 1e-6:
                    z = centroid[2] - (normal[0] * (xx - centroid[0]) + normal[1] * (yy - centroid[1])) / normal[2]
                    self.ax6.plot_surface(xx, yy, z, alpha=0.5, color='cyan')
                    
        self.ax6.set_title("3D Point Cloud & Plane Fit")
        self.ax6.set_xlabel('X')
        self.ax6.set_ylabel('Y')
        self.ax6.set_zlabel('Depth (Z)')
        
        self.fig.suptitle(f"Solar Panel Pose Estimator | Pitch: {est_pitch:.1f}° | Roll: {est_roll:.1f}° | Error: {np.mean([abs(err_pitch), abs(err_roll)]):.2f}°", fontsize=14)
        
        plt.pause(0.1)
