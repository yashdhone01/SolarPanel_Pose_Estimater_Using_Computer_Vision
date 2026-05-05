import cv2
import numpy as np
import simulate
import depth
import sift_match
import plane_fit
import hazard
import failure

class VisualServoSystem:
    def __init__(self):
        self.calib_data = np.load('calibration_data.npz')
        self.P1 = self.calib_data['P1']
        self.f = self.P1[0, 0]
        self.cx = self.P1[0, 2]
        self.cy = self.P1[1, 2]
        self.B = self.calib_data['B']
        
        self.target_pixel = None
        self.target_3D = None
        
        # Lander State [X, Y, Z]
        # Start hovering 1.5 meters from the camera
        self.lander_pos = np.array([0.0, 0.0, 1.5])
        self.lander_tilt = 0.0
        
        # Control gain for Proportional Control
        self.Kp_pos = 0.05
        self.Kp_tilt = 0.1
        
        cv2.namedWindow('Visual Servoing')
        cv2.setMouseCallback('Visual Servoing', self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.target_pixel = (x, y)
            
    def get_valid_disparity(self, disp_map, u, v, search_radius=5):
        """
        Robustly searches a local neighborhood for a valid disparity 
        if the exact clicked pixel falls into an occlusion or noise gap.
        """
        h, w = disp_map.shape
        # Search outward in expanding rings
        for r in range(search_radius + 1):
            for i in range(-r, r+1):
                for j in range(-r, r+1):
                    # Only check the perimeter of the current ring
                    if max(abs(i), abs(j)) == r:
                        nu, nv = u + i, v + j
                        if 0 <= nu < w and 0 <= nv < h:
                            d = disp_map[nv, nu]
                            if d > 0:
                                return d
        return None

    def unproject(self, u, v, d):
        """
        Explicit geometric unprojection from 2D + Disparity to 3D Physical Coordinates.
        """
        Z = (self.f * self.B) / d
        X = (u - self.cx) * Z / self.f
        Y = (v - self.cy) * Z / self.f
        return np.array([X, Y, Z])
        
    def project(self, X, Y, Z):
        """
        Explicit geometric projection from 3D Physical Coordinates back to 2D Screen.
        """
        if Z <= 0: return None
        u = int((X * self.f / Z) + self.cx)
        v = int((Y * self.f / Z) + self.cy)
        return (u, v)

    def run(self):
        print("--- Real-time Visual Servoing System Active ---")
        
        # Simulation background parameters
        sim_z = 2.5
        sim_p = 0.0
        sim_r = 0.0
        
        sun_vector = np.array([0, -np.sin(np.radians(45)), np.cos(np.radians(45))])
        
        while True:
            # 1. Generate stereo frame (simulate the real-world feed)
            img_l, img_r, _ = simulate.generate_stereo_frame(sim_p, sim_r, z_dist=sim_z, add_noise_flag=True)
            
            # 2. Rectify
            img_l_rect = cv2.remap(img_l, self.calib_data['map1_L'], self.calib_data['map2_L'], cv2.INTER_LINEAR)
            img_r_rect = cv2.remap(img_r, self.calib_data['map1_R'], self.calib_data['map2_R'], cv2.INTER_LINEAR)
            
            # 3. SIFT & Masking
            sift_result = sift_match.match_and_mask(img_l_rect, img_r_rect)
            
            status = "NO RELIABLE ESTIMATE"
            confidence = "LOW"
            tilt = 0.0
            error_dist = 0.0
            
            display_img = cv2.cvtColor(img_l_rect, cv2.COLOR_GRAY2BGR)
            
            if sift_result[0] is not None:
                inlier_kp, match_img, H, panel_mask = sift_result
                
                # We need raw disparity to unproject the user's specific click
                block_size = 11
                sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=block_size, P1=8*3*block_size**2, P2=32*3*block_size**2)
                disp_16 = sgbm.compute(img_l_rect, img_r_rect)
                disp = disp_16.astype(np.float32) / 16.0
                
                # 4. Target Acquisition & Tracking (User Click -> 3D)
                if self.target_pixel is not None:
                    u, v = self.target_pixel
                    d = self.get_valid_disparity(disp, u, v, search_radius=5)
                    if d is not None:
                        self.target_3D = self.unproject(u, v, d)
                        # Consume the click, target_3D persists
                        self.target_pixel = None
                        
                # 5. Pipeline Analytics (Tilt, Confidence, Hazards)
                _, _, _, point_cloud, _, _, valid_pts, z_std = depth.compute_depth_and_pointcloud(img_l, img_r, self.calib_data, panel_mask)
                
                normal, est_pitch, est_roll, mean_residual, centroid = plane_fit.fit_plane(point_cloud)
                if normal is not None:
                    tilt = est_pitch
                    n = normal / np.linalg.norm(normal)
                    s = sun_vector / np.linalg.norm(sun_vector)
                    align_err = np.degrees(np.arccos(np.clip(np.dot(n, s), -1.0, 1.0)))
                    
                    inlier_count = sum(inlier_kp[3])
                    status, confidence = failure.evaluate_confidence(inlier_count, valid_pts, z_std, align_err)
                    
                    # 6. CLOSED-LOOP PROPORTIONAL CONTROL
                    if self.target_3D is not None:
                        # Error vector
                        error_vec = self.target_3D - self.lander_pos
                        error_dist = np.linalg.norm(error_vec) * 100.0 # in cm
                        
                        # Kinematic Proportional Velocity Update: V = Kp * E
                        velocity = self.Kp_pos * error_vec
                        self.lander_pos += velocity
                        
                        # Auto-align lander tilt using proportional control
                        # It tracks the surface tilt, then we simulate correcting it
                        self.lander_tilt += self.Kp_tilt * (tilt - self.lander_tilt)
                        
                        # 7. PROJECTION & VISUALIZATION
                        t_uv = self.project(*self.target_3D)
                        l_uv = self.project(*self.lander_pos)
                        
                        if l_uv is not None and t_uv is not None:
                            t_u, t_v = t_uv
                            l_u, l_v = l_uv
                            
                            # Draw Path
                            cv2.line(display_img, (l_u, l_v), (t_u, t_v), (255, 255, 0), 2)
                            
                            # Draw Target
                            cv2.circle(display_img, (t_u, t_v), 10, (0, 0, 255), 2)
                            cv2.putText(display_img, "TARGET", (t_u+15, t_v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                            
                            # Draw Lander (Dynamic scaling based on Z would be cool, but fixed marker works)
                            # Let's scale marker size based on altitude
                            marker_size = max(5, int(40 / self.lander_pos[2]))
                            cv2.drawMarker(display_img, (l_u, l_v), (0, 255, 0), cv2.MARKER_SQUARE, marker_size, 3)
                            cv2.putText(display_img, "LANDER", (l_u-40, l_v-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                            
                            # Landing Logic evaluation
                            if error_dist < 5.0 and abs(self.lander_tilt) < 5.0 and confidence == 'HIGH':
                                status = "LANDING SUCCESS"
                            elif abs(tilt) > 15.0 or confidence == 'LOW':
                                status = "UNSTABLE / ABORT"
                                
            # 8. HUD OVERLAY
            hud_y = 30
            cv2.putText(display_img, "VISUAL SERVOING SYSTEM", (20, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2); hud_y+=30
            cv2.putText(display_img, f"Target 3D Tracked: {self.target_3D is not None}", (20, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1); hud_y+=30
            cv2.putText(display_img, f"Lander Altitude: {self.lander_pos[2]*100:.1f} cm", (20, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1); hud_y+=30
            cv2.putText(display_img, f"Error Dist: {error_dist:.1f} cm", (20, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1); hud_y+=30
            cv2.putText(display_img, f"Surface Tilt: {tilt:.1f} deg", (20, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1); hud_y+=30
            
            c_color = (0, 255, 0) if confidence == 'HIGH' else (0, 0, 255)
            cv2.putText(display_img, f"Confidence: {confidence}", (20, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_color, 2); hud_y+=30
            
            s_color = (0, 255, 0) if "SUCCESS" in status else ((0, 0, 255) if "ABORT" in status else (255, 255, 255))
            cv2.putText(display_img, f"Status: {status}", (20, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, s_color, 2); hud_y+=30
            
            cv2.putText(display_img, "[Click] to set target | [Q] to quit", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow('Visual Servoing', display_img)
            
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = VisualServoSystem()
    system.run()
