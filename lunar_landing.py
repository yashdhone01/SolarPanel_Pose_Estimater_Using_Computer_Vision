import numpy as np
import cv2
import calibrate
import simulate
import sift_match
import depth
import plane_fit
import hud
import visualization
import controls
import failure
import game
import hazard

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
    
    normal_vector = np.array([0., 0., 1.])
    metrics = {
        'altitude': 0.0,
        'tilt': 0.0,
        'error': 0.0,
        'confidence': 'LOW',
        'hazard': False
    }
    
    print("--- Lunar Landing Pipeline Active ---")
    cv2.namedWindow('Dashboard', cv2.WINDOW_NORMAL)
    controls.setup_mouse('Dashboard')
    
    game_state = game.LanderState()
    
    # Initialize buffers
    img_left_rect = None
    disp_vis = None
    match_img = None
    
    while True:
        # Check mouse input
        click = controls.get_mouse_click()
        if click:
            game_state.set_target(click[0], click[1])
            
        key = controls.process_input(delay=10)
        
        if key == 'q':
            break
        elif key == 'left':
            sun_azimuth -= 0.1
        elif key == 'right':
            sun_azimuth += 0.1
        elif key == 'up':
            sun_elevation += 0.1
        elif key == 'down':
            sun_elevation -= 0.1
        elif key == 'w':
            game_state.input_thrust(-0.5) # decrease speed (thrust up)
        elif key == 's':
            game_state.input_thrust(0.5) # increase speed (thrust down)
        elif key == 'a':
            game_state.input_lateral(-5, 0)
        elif key == 'd':
            game_state.input_lateral(5, 0)
        elif key == 'r':
            game_state.reset()
            sun_azimuth = 0.0
            sun_elevation = np.radians(45)
            metrics = {k: 0 for k in metrics if k not in ['hazard']}
            metrics['confidence'] = 'LOW'
            metrics['hazard'] = False
            img_left_rect, disp_vis, match_img = None, None, None
            
        sun_vector = normalize(get_sun_vector(sun_azimuth, sun_elevation))
        
        # Determine panel orientation:
        # The panel tries to track the sun, but user must "SPACE" to snap it.
        if key == 'space':
            p = np.degrees(np.arcsin(np.clip(sun_vector[1], -1.0, 1.0)))
            r_cos = np.cos(np.radians(p))
            if abs(r_cos) > 1e-6:
                r = np.degrees(np.arcsin(np.clip(sun_vector[0] / r_cos, -1.0, 1.0)))
            else:
                r = 0.0
            game_state.panel_p = p
            game_state.panel_r = r
            
        if not hasattr(game_state, 'panel_p'):
            game_state.panel_p = 0.0
            game_state.panel_r = 0.0
            
        # --- CONTINUOUS CV LOOP ---
        gt_z = game_state.true_altitude_cm / 100.0
        if gt_z < 0.1: gt_z = 0.1
        
        p = game_state.panel_p + np.random.normal(0, 0.5)
        r = game_state.panel_r + np.random.normal(0, 0.5)
        
        # Only run CV if not frozen (Success/Crash)
        if not game_state.frozen:
            img_l, img_r, gt = simulate.generate_stereo_frame(p, r, z_dist=gt_z, add_noise_flag=True)
            
            map1_L = calib_data['map1_L']
            map2_L = calib_data['map2_L']
            map1_R = calib_data['map1_R']
            map2_R = calib_data['map2_R']
            
            img_left_rect = cv2.remap(img_l, map1_L, map2_L, cv2.INTER_LINEAR)
            img_r_rect = cv2.remap(img_r, map1_R, map2_R, cv2.INTER_LINEAR)
            
            sift_result = sift_match.match_and_mask(img_left_rect, img_r_rect)
            
            if sift_result[0] is not None:
                inlier_keypoints, match_img, H, panel_mask = sift_result
                _, _, disp_vis, point_cloud, center_depth, center_pt, valid_points, z_std = depth.compute_depth_and_pointcloud(img_l, img_r, calib_data, panel_mask)
                
                metrics['center_depth'] = center_depth * 100.0 # cm
                metrics['center_pt'] = center_pt
                
                kp_l, kp_r, good_matches, matchesMask = inlier_keypoints
                inlier_count = sum(matchesMask)
                
                img_left_rect = cv2.cvtColor(img_left_rect, cv2.COLOR_GRAY2BGR)
                img_left_rect = cv2.drawKeypoints(img_left_rect, kp_l, None, color=(0,255,0), flags=0)
                
                normal, est_pitch, est_roll, mean_residual, centroid = plane_fit.fit_plane(point_cloud)
                
                if normal is not None:
                    normal_vector = normal
                    metrics['altitude'] = centroid[2] * 100.0
                    metrics['tilt'] = est_pitch
                    
                    n = normal / np.linalg.norm(normal)
                    s = sun_vector / np.linalg.norm(sun_vector)
                    angle_deg = np.degrees(np.arccos(np.clip(np.dot(n, s), -1.0, 1.0)))
                    metrics['error'] = angle_deg
                    
                    status, confidence = failure.evaluate_confidence(inlier_count, valid_points, z_std, angle_deg)
                    metrics['status'] = status
                    metrics['confidence'] = confidence
                    
                    is_hazard = hazard.check_hazard(mean_residual)
                    metrics['hazard'] = is_hazard
                    
                    # Update Game State
                    game_state.update(est_pitch, angle_deg, confidence, is_hazard, metrics['center_depth'])
                else:
                    metrics['status'] = 'NO RELIABLE ESTIMATE'
                    metrics['confidence'] = 'LOW'
                    game_state.update(999.0, 999.0, 'LOW', True, 0.0)
            else:
                metrics['status'] = 'LOW FEATURE CONFIDENCE'
                metrics['confidence'] = 'LOW'
                game_state.update(999.0, 999.0, 'LOW', True, 0.0)

        # Render 3D
        img_3d = visualization.get_3d_render(normal_vector, sun_vector)
        
        # Build Dashboard
        dashboard = hud.build_dashboard(img_left_rect, disp_vis, match_img, img_3d, game_state, metrics)
        cv2.imshow('Dashboard', dashboard)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
