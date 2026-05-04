import cv2
import numpy as np

def build_dashboard(img_left, img_disp, img_matches, img_3d, metrics):
    """
    Composites the 4 panels and overlays HUD telemetry.
    """
    target_size = (640, 480)
    
    # Extract center point for crosshair
    center_pt = metrics.get('center_pt', (0, 0))
    valid_crosshair = center_pt != (0, 0)
    
    # Helper to draw crosshair
    def draw_crosshair(img, pt):
        if not valid_crosshair: return img
        # Scale pt to target_size if original was different (original is likely 640x480 anyway)
        # Assuming original is 640x480 for synthetic stereo
        cx, cy = pt
        cv2.line(img, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 2)
        cv2.line(img, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 2)
        cv2.circle(img, (cx, cy), 3, (0, 255, 255), -1)
        return img

    # 1. Top Left: Live Camera (img_left)
    if img_left is None:
        img_left_c = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        if len(img_left.shape) == 2:
            img_left_c = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
        else:
            img_left_c = img_left.copy()
            
    img_left_c = cv2.resize(img_left_c, target_size)
    img_left_c = draw_crosshair(img_left_c, center_pt)
    cv2.putText(img_left_c, "LEFT VIEW (RECTIFIED)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # 2. Top Right: Disparity Heatmap
    if img_disp is None:
        img_disp_c = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        # disp_vis is already colored now via depth.py
        img_disp_c = img_disp.copy()
        
    img_disp_c = cv2.resize(img_disp_c, target_size)
    img_disp_c = draw_crosshair(img_disp_c, center_pt)
    cv2.putText(img_disp_c, "DEPTH MAP (DISPARITY)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # 3. Bottom Left: SIFT Matches
    if img_matches is None:
        img_matches_c = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        img_matches_c = cv2.resize(img_matches, target_size)
    cv2.putText(img_matches_c, "SIFT Stereo Matches", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # 4. Bottom Right: 3D Visualization
    if img_3d is None:
        img_3d_c = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        img_3d_c = cv2.resize(img_3d, target_size)
    
    # Assemble 2x2 grid
    top_row = np.hstack((img_left_c, img_disp_c))
    bottom_row = np.hstack((img_matches_c, img_3d_c))
    dashboard = np.vstack((top_row, bottom_row))
    
    # Draw Main HUD overlay
    hud_overlay = dashboard.copy()
    hud_rect_x, hud_rect_y, hud_rect_w, hud_rect_h = 20, 60, 450, 300
    cv2.rectangle(hud_overlay, (hud_rect_x, hud_rect_y), (hud_rect_x + hud_rect_w, hud_rect_y + hud_rect_h), (0, 0, 0), -1)
    
    # Alpha blend overlay
    alpha = 0.6
    cv2.addWeighted(hud_overlay, alpha, dashboard, 1 - alpha, 0, dashboard)
    
    # Draw HUD Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(dashboard, "SYSTEM TELEMETRY", (30, 100), font, 0.8, (255, 255, 0), 2)
    
    depth_val = metrics.get('center_depth', 0.0)
    cv2.putText(dashboard, f"Depth at center: {depth_val:.1f} cm", (30, 145), font, 0.7, (0, 255, 255), 2)
    
    alt = metrics.get('altitude', 0.0)
    cv2.putText(dashboard, f"Altitude:        {alt:.1f} cm", (30, 185), font, 0.7, (255, 255, 255), 2)
    
    tilt = metrics.get('tilt', 0.0)
    cv2.putText(dashboard, f"Tilt Angle:      {tilt:.1f} deg", (30, 225), font, 0.7, (255, 255, 255), 2)
    
    err = metrics.get('error', 0.0)
    cv2.putText(dashboard, f"Align Error:     {err:.1f} deg", (30, 265), font, 0.7, (255, 255, 255), 2)
    
    status_text = metrics.get('status', 'WAITING')
    status_color = (0, 255, 0) if status_text == "ALIGNED" else (0, 0, 255)
    cv2.putText(dashboard, f"STATUS:          {status_text}", (30, 305), font, 0.7, status_color, 2)
    
    conf = metrics.get('confidence', 'LOW')
    conf_color = (0, 255, 0) if conf == 'HIGH' else (0, 0, 255)
    cv2.putText(dashboard, f"CONFIDENCE:      {conf}", (30, 345), font, 0.7, conf_color, 2)
    
    # Draw Controls Overlay
    ctrl_overlay = dashboard.copy()
    cv2.rectangle(ctrl_overlay, (20, 960 - 160), (320, 960 - 20), (0, 0, 0), -1)
    cv2.addWeighted(ctrl_overlay, alpha, dashboard, 1 - alpha, 0, dashboard)
    
    cv2.putText(dashboard, "CONTROLS", (30, 960 - 120), font, 0.7, (255, 255, 0), 2)
    cv2.putText(dashboard, "[Arrows] Move Sun", (30, 960 - 90), font, 0.6, (255, 255, 255), 1)
    cv2.putText(dashboard, "[SPACE]  Lock Track", (30, 960 - 65), font, 0.6, (255, 255, 255), 1)
    cv2.putText(dashboard, "[R]      Reset", (30, 960 - 40), font, 0.6, (255, 255, 255), 1)
    
    return dashboard
