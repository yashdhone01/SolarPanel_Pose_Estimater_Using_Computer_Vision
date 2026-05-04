import numpy as np
import cv2
import calibrate
import simulate
import sift_match
import depth
import plane_fit
import interactive_display
import time
import matplotlib.pyplot as plt

def main():
    # 1. Run Calibration (caches to npz)
    calibrate.run_calibration()
    
    # Load calibration data
    calib_data = np.load('calibration_data.npz')
    
    interactive_display.init_display()
    
    # State flags
    sun_azimuth = 0.0
    sun_elevation = np.radians(45)
    
    # Defaults
    def get_sun_vector(az, el):
        return np.array([np.cos(el)*np.sin(az), -np.sin(el), np.cos(el)*np.cos(az)])
    
    def normalize(v):
        return v / np.linalg.norm(v)
        
    sun_vector = normalize(get_sun_vector(sun_azimuth, sun_elevation))

    print("System ready. Controls:")
    print("  Arrow Keys: Move Sun")
    print("  SPACE     : Trigger full CV Pipeline update")
    print("  r         : Reset state")
    print("  q         : Quit")
    
    # Generate initial state visually
    interactive_display.update_sun_only(sun_vector)
    
    while True:
        key = interactive_display.get_key()
        if key is not None:
            if key == 'left':
                sun_azimuth -= 0.1
                sun_vector = normalize(get_sun_vector(sun_azimuth, sun_elevation))
                interactive_display.update_sun_only(sun_vector)
            elif key == 'right':
                sun_azimuth += 0.1
                sun_vector = normalize(get_sun_vector(sun_azimuth, sun_elevation))
                interactive_display.update_sun_only(sun_vector)
            elif key == 'up':
                sun_elevation += 0.1
                sun_vector = normalize(get_sun_vector(sun_azimuth, sun_elevation))
                interactive_display.update_sun_only(sun_vector)
            elif key == 'down':
                sun_elevation -= 0.1
                # Limit elevation so sun doesn't go below horizon if desired, but we'll let it freely orbit
                sun_vector = normalize(get_sun_vector(sun_azimuth, sun_elevation))
                interactive_display.update_sun_only(sun_vector)
            elif key == 'r':
                sun_azimuth = 0.0
                sun_elevation = np.radians(45)
                sun_vector = normalize(get_sun_vector(sun_azimuth, sun_elevation))
                interactive_display.update_sun_only(sun_vector)
                print("Reset sun.")
            elif key == 'q':
                break
            elif key == ' ':
                print("--- Triggering CV Pipeline (Solar Tracking Mode) ---")
                
                # Result-Oriented Logic: The panel aligns to track the sun!
                gt_z = np.random.uniform(2.0, 3.0)
                
                # Calculate required pitch and roll to match the sun_vector
                p = np.degrees(np.arcsin(np.clip(sun_vector[1], -1.0, 1.0)))
                r_cos = np.cos(np.radians(p))
                if abs(r_cos) > 1e-6:
                    r = np.degrees(np.arcsin(np.clip(sun_vector[0] / r_cos, -1.0, 1.0)))
                else:
                    r = 0.0
                    
                # Add slight physical imperfection (e.g., servo motor inaccuracy 0.5 degrees)
                p += np.random.normal(0, 0.5)
                r += np.random.normal(0, 0.5)
                
                img_l, img_r, gt = simulate.generate_stereo_frame(p, r, z_dist=gt_z, add_noise_flag=True)
                
                map1_L = calib_data['map1_L']
                map2_L = calib_data['map2_L']
                map1_R = calib_data['map1_R']
                map2_R = calib_data['map2_R']
                
                img_l_rect = cv2.remap(img_l, map1_L, map2_L, cv2.INTER_LINEAR)
                img_r_rect = cv2.remap(img_r, map1_R, map2_R, cv2.INTER_LINEAR)
                
                sift_result = sift_match.match_and_mask(img_l_rect, img_r_rect)
                if sift_result[0] is None:
                    print("WARNING: SIFT failed.")
                    continue
                    
                inlier_keypoints, match_img, H, panel_mask = sift_result
                img_l_rect_dummy, img_r_rect_dummy, disp_vis, point_cloud = depth.compute_depth_and_pointcloud(img_l, img_r, calib_data, panel_mask)
                normal, est_pitch, est_roll, mean_residual, centroid = plane_fit.fit_plane(point_cloud)
                
                if normal is None:
                    print("WARNING: Plane fit failed.")
                    continue
                    
                data_dict = {
                    'left_img': img_l_rect,
                    'right_img': img_r_rect,
                    'matches_img': match_img,
                    'disparity_map': disp_vis,
                    'estimated_pitch_roll': (est_pitch, est_roll),
                    'ground_truth_pitch_roll': (gt['pitch'], gt['roll']),
                    'normal_vector': normal,
                    'sun_vector': sun_vector
                }
                interactive_display.update_full_display(data_dict)
                
        plt.pause(0.01)

if __name__ == "__main__":
    main()
