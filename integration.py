import numpy as np
import cv2
import simulate
import sift_match
import depth
import plane_fit
import hazard
import failure

class CVSystem:
    def __init__(self):
        self.calib_data = np.load('calibration_data.npz')
        self.map1_L = self.calib_data['map1_L']
        self.map2_L = self.calib_data['map2_L']
        self.map1_R = self.calib_data['map1_R']
        self.map2_R = self.calib_data['map2_R']

    def get_sun_vector(self, az, el):
        return np.array([np.cos(el)*np.sin(az), -np.sin(el), np.cos(el)*np.cos(az)])

    def normalize(self, v):
        n = np.linalg.norm(v)
        if n == 0: return v
        return v / n

    def update_from_cv(self, gt_z_m, sun_azimuth, sun_elevation, panel_p, panel_r):
        """
        Runs the full CV pipeline and explicitly returns physics values.
        """
        if gt_z_m < 0.1: gt_z_m = 0.1
        
        sun_vector = self.normalize(self.get_sun_vector(sun_azimuth, sun_elevation))
        
        # Add slight noise to base targets
        p = panel_p + np.random.normal(0, 0.5)
        r = panel_r + np.random.normal(0, 0.5)
        
        img_l, img_r, gt = simulate.generate_stereo_frame(p, r, z_dist=gt_z_m, add_noise_flag=True)
        
        img_left_rect = cv2.remap(img_l, self.map1_L, self.map2_L, cv2.INTER_LINEAR)
        img_r_rect = cv2.remap(img_r, self.map1_R, self.map2_R, cv2.INTER_LINEAR)
        
        sift_result = sift_match.match_and_mask(img_left_rect, img_r_rect)
        
        # Default fail values
        depth_Z = gt_z_m * 100.0 # fallback to simulation altitude
        tilt_angle = 0.0
        alignment_error = 999.0
        confidence = "LOW"
        is_hazard = True
        normal_vector = np.array([0, 0, 1])
        status = "NO RELIABLE ESTIMATE"
        
        if sift_result[0] is not None:
            inlier_keypoints, match_img, H, panel_mask = sift_result
            _, _, _, point_cloud, center_depth, center_pt, valid_points, z_std = depth.compute_depth_and_pointcloud(img_l, img_r, self.calib_data, panel_mask)
            
            if center_depth > 0:
                depth_Z = center_depth * 100.0 # to cm
                
            kp_l, kp_r, good_matches, matchesMask = inlier_keypoints
            inlier_count = sum(matchesMask)
            
            normal, est_pitch, est_roll, mean_residual, centroid = plane_fit.fit_plane(point_cloud)
            if normal is not None:
                normal_vector = normal
                tilt_angle = est_pitch # Map CV tilt to pitch
                
                n = normal / np.linalg.norm(normal)
                s = sun_vector / np.linalg.norm(sun_vector)
                alignment_error = np.degrees(np.arccos(np.clip(np.dot(n, s), -1.0, 1.0)))
                
                status, confidence = failure.evaluate_confidence(inlier_count, valid_points, z_std, alignment_error)
                is_hazard = hazard.check_hazard(mean_residual)

        return {
            'depth_Z': depth_Z,
            'tilt_angle': tilt_angle,
            'alignment_error': alignment_error,
            'confidence': confidence,
            'status': status,
            'hazard': is_hazard,
            'normal': normal_vector
        }
