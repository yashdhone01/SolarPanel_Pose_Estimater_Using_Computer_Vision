import numpy as np
import cv2
import calibrate
import simulate
import sift_match
import depth
import plane_fit
import ui
import visualization
import controls

def main():
    calibrate.run_calibration()
    calib_data = np.load('calibration_data.npz')
    
    sun_azimuth = 0.0
    sun_elevation = np.radians(45)
    
    def get_sun_vector(az, el):
        return np.array([np.cos(el)*np.sin(az), -np.sin(el), np.cos(el)*np.cos(az)])
    
    def normalize(v):
        return v / np.linalg.norm(v)
        
    sun_vector = normalize(get_sun_vector(sun_azimuth, sun_elevation))
    
    # Init empty buffers for live tracking
    img_left_rect = None
    disp_vis = None
    match_img = None
    normal_vector = np.array([0., 0., 1.])
    
    metrics = {
        'altitude': 0.0,
        'tilt': 0.0,
        'error': 0.0,
        'confidence': 'LOW'
    }
    
    print("--- 3D Solar Panel Pipeline Active ---")
    cv2.namedWindow('Dashboard', cv2.WINDOW_NORMAL)
    
    while True:
        # Render 3D view natively off-screen to image
        img_3d = visualization.get_3d_render(normal_vector, sun_vector)
        
        # Build layout
        dashboard = ui.build_dashboard(img_left_rect, disp_vis, match_img, img_3d, metrics)
        
        # Render
        cv2.imshow('Dashboard', dashboard)
        
        key = controls.process_input(delay=20) # 20ms = ~50 FPS
        
        if key == 'q':
            break
        elif key == 'left':
            sun_azimuth -= 0.1
            sun_vector = normalize(get_sun_vector(sun_azimuth, sun_elevation))
        elif key == 'right':
            sun_azimuth += 0.1
            sun_vector = normalize(get_sun_vector(sun_azimuth, sun_elevation))
        elif key == 'up':
            sun_elevation += 0.1
            sun_vector = normalize(get_sun_vector(sun_azimuth, sun_elevation))
        elif key == 'down':
            sun_elevation -= 0.1
            sun_vector = normalize(get_sun_vector(sun_azimuth, sun_elevation))
        elif key == 'r':
            sun_azimuth = 0.0
            sun_elevation = np.radians(45)
            sun_vector = normalize(get_sun_vector(sun_azimuth, sun_elevation))
            normal_vector = np.array([0., 0., 1.])
            metrics = {k: 0 for k in metrics}
            metrics['confidence'] = 'LOW'
            img_left_rect, disp_vis, match_img = None, None, None
        elif key == 'space':
            # CV Pipeline Execution
            gt_z = np.random.uniform(2.0, 3.0)
            
            p = np.degrees(np.arcsin(np.clip(sun_vector[1], -1.0, 1.0)))
            r_cos = np.cos(np.radians(p))
            if abs(r_cos) > 1e-6:
                r = np.degrees(np.arcsin(np.clip(sun_vector[0] / r_cos, -1.0, 1.0)))
            else:
                r = 0.0
                
            p += np.random.normal(0, 0.5)
            r += np.random.normal(0, 0.5)
            
            img_l, img_r, gt = simulate.generate_stereo_frame(p, r, z_dist=gt_z, add_noise_flag=True)
            
            map1_L = calib_data['map1_L']
            map2_L = calib_data['map2_L']
            map1_R = calib_data['map1_R']
            map2_R = calib_data['map2_R']
            
            img_left_rect = cv2.remap(img_l, map1_L, map2_L, cv2.INTER_LINEAR)
            img_r_rect = cv2.remap(img_r, map1_R, map2_R, cv2.INTER_LINEAR)
            
            sift_result = sift_match.match_and_mask(img_left_rect, img_r_rect)
            if sift_result[0] is None:
                metrics['confidence'] = 'LOW'
                continue
                
            inlier_keypoints, match_img, H, panel_mask = sift_result
            _, _, disp_vis, point_cloud = depth.compute_depth_and_pointcloud(img_l, img_r, calib_data, panel_mask)
            
            # SIFT points drawn on left image
            kp_l, kp_r, good_matches, matchesMask = inlier_keypoints
            inlier_count = sum(matchesMask)
            inlier_ratio = inlier_count / max(1, len(good_matches))
            metrics['confidence'] = 'HIGH' if inlier_ratio > 0.5 else 'LOW'
            
            # Draw keypoints on left rect
            img_left_rect = cv2.cvtColor(img_left_rect, cv2.COLOR_GRAY2BGR)
            img_left_rect = cv2.drawKeypoints(img_left_rect, kp_l, None, color=(0,255,0), flags=0)
            
            normal, est_pitch, est_roll, mean_residual, centroid = plane_fit.fit_plane(point_cloud)
            if normal is not None:
                normal_vector = normal
                metrics['altitude'] = centroid[2] * 100.0 # to cm
                metrics['tilt'] = est_pitch
                
                n = normal / np.linalg.norm(normal)
                s = sun_vector / np.linalg.norm(sun_vector)
                angle_deg = np.degrees(np.arccos(np.clip(np.dot(n, s), -1.0, 1.0)))
                metrics['error'] = angle_deg

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
