import cv2
import numpy as np
import random

def build_dashboard(img_left, img_disp, img_matches, img_3d, game_state, metrics):
    """
    Composites the 4 panels and overlays Lunar Landing Guidance telemetry.
    """
    target_size = (640, 480)
    
    center_pt = metrics.get('center_pt', (0, 0))
    valid_crosshair = center_pt != (0, 0)
    
    def draw_crosshair(img, pt, color=(0,255,0)):
        if not valid_crosshair: return img
        cx, cy = pt
        cv2.line(img, (cx - 10, cy), (cx + 10, cy), color, 2)
        cv2.line(img, (cx, cy - 10), (cx, cy + 10), color, 2)
        cv2.circle(img, (cx, cy), 3, (0, 255, 255), -1)
        return img
        
    def draw_target(img, tx, ty, lander_x, lander_y, hazard):
        # Draw Target Zone
        lz_color = (0, 0, 255) if hazard else (0, 255, 0)
        cv2.circle(img, (tx, ty), 15, lz_color, 2)
        cv2.putText(img, "LZ", (tx + 20, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lz_color, 1)
        
        # Draw Lander
        cv2.drawMarker(img, (int(lander_x), int(lander_y)), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
        return img

    # 1. Top Left: Live Camera (img_left)
    if img_left is None: img_left_c = np.zeros((480, 640, 3), dtype=np.uint8)
    else: img_left_c = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR) if len(img_left.shape) == 2 else img_left.copy()
    img_left_c = cv2.resize(img_left_c, target_size)
    img_left_c = draw_crosshair(img_left_c, center_pt)
    hazard = metrics.get('hazard', False)
    img_left_c = draw_target(img_left_c, game_state.target_x, game_state.target_y, game_state.lander_x, game_state.lander_y, hazard)
    cv2.putText(img_left_c, "LEFT VIEW (RECTIFIED)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # 2. Top Right: Disparity Heatmap
    if img_disp is None: img_disp_c = np.zeros((480, 640, 3), dtype=np.uint8)
    else: img_disp_c = cv2.resize(img_disp.copy(), target_size)
    img_disp_c = draw_crosshair(img_disp_c, center_pt)
    cv2.putText(img_disp_c, "DEPTH MAP (DISPARITY)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # 3. Bottom Left: SIFT Matches
    if img_matches is None: img_matches_c = np.zeros((480, 640, 3), dtype=np.uint8)
    else: img_matches_c = cv2.resize(img_matches, target_size)
    cv2.putText(img_matches_c, "SIFT Stereo Matches", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # 4. Bottom Right: 3D Visualization
    if img_3d is None: img_3d_c = np.zeros((480, 640, 3), dtype=np.uint8)
    else: img_3d_c = cv2.resize(img_3d, target_size)
    
    # Assemble 2x2 grid
    top_row = np.hstack((img_left_c, img_disp_c))
    bottom_row = np.hstack((img_matches_c, img_3d_c))
    dashboard = np.vstack((top_row, bottom_row))
    
    # Draw Main HUD overlay
    hud_overlay = dashboard.copy()
    hud_rect_x, hud_rect_y, hud_rect_w, hud_rect_h = 20, 60, 450, 420
    cv2.rectangle(hud_overlay, (hud_rect_x, hud_rect_y), (hud_rect_x + hud_rect_w, hud_rect_y + hud_rect_h), (0, 0, 0), -1)
    
    alpha = 0.6
    cv2.addWeighted(hud_overlay, alpha, dashboard, 1 - alpha, 0, dashboard)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(dashboard, "LUNAR LANDING TELEMETRY", (30, 100), font, 0.8, (255, 255, 0), 2)
    
    cv2.putText(dashboard, f"Altitude:        {game_state.true_altitude_cm:.1f} cm", (30, 145), font, 0.7, (255, 255, 255), 2)
    cv2.putText(dashboard, f"Lateral Error:   {game_state.error_cm:.1f} cm", (30, 185), font, 0.7, (255, 255, 255), 2)
    cv2.putText(dashboard, f"Tilt Angle:      {metrics.get('tilt', 0.0):.1f} deg", (30, 225), font, 0.7, (255, 255, 255), 2)
    
    # Guidance
    cv2.putText(dashboard, f"GUIDANCE:        {game_state.guidance_msg}", (30, 265), font, 0.7, game_state.guidance_color, 2)
    
    # Confidence and Status
    conf = metrics.get('confidence', 'LOW')
    conf_color = (0, 255, 0) if conf == 'HIGH' else (0, 0, 255)
    cv2.putText(dashboard, f"CONFIDENCE:      {conf}", (30, 305), font, 0.7, conf_color, 2)
    
    status_text = metrics.get('status', 'WAITING')
    status_color = (0, 255, 0) if status_text == "ALIGNED" else (0, 0, 255)
    cv2.putText(dashboard, f"CV STATUS:       {status_text}", (30, 345), font, 0.7, status_color, 2)
    
    depth_val = metrics.get('center_depth', 0.0)
    cv2.putText(dashboard, f"Depth@Center:    {depth_val:.1f} cm", (30, 385), font, 0.7, (0, 255, 255), 2)
    
    hz_str = "HAZARD DETECTED" if hazard else "SAFE LZ"
    hz_col = (0, 0, 255) if hazard else (0, 255, 0)
    cv2.putText(dashboard, f"LZ GEOMETRY:     {hz_str}", (30, 425), font, 0.7, hz_col, 2)
    
    # Draw Controls Overlay
    ctrl_overlay = dashboard.copy()
    cv2.rectangle(ctrl_overlay, (20, 960 - 200), (380, 960 - 20), (0, 0, 0), -1)
    cv2.addWeighted(ctrl_overlay, alpha, dashboard, 1 - alpha, 0, dashboard)
    
    cv2.putText(dashboard, "CONTROLS", (30, 960 - 160), font, 0.7, (255, 255, 0), 2)
    cv2.putText(dashboard, "[Arrows] Move Sun Vector", (30, 960 - 130), font, 0.6, (255, 255, 255), 1)
    cv2.putText(dashboard, "[W/S]    Thrust (Alt)", (30, 960 - 105), font, 0.6, (255, 255, 255), 1)
    cv2.putText(dashboard, "[A/D]    Lateral Drift", (30, 960 - 80), font, 0.6, (255, 255, 255), 1)
    cv2.putText(dashboard, "[SPACE]  Auto-Align", (30, 960 - 55), font, 0.6, (255, 255, 255), 1)
    cv2.putText(dashboard, "[Mouse]  Select LZ", (30, 960 - 30), font, 0.6, (255, 255, 255), 1)
    
    # End Game Animations
    if game_state.state == 'SUCCESS':
        overlay = dashboard.copy()
        cv2.rectangle(overlay, (0,0), (1280, 960), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, dashboard, 0.7, 0, dashboard)
        cv2.putText(dashboard, "TOUCHDOWN SUCCESS", (300, 480), font, 2.0, (255, 255, 255), 5)
        # Simple confetti
        for _ in range(200):
            x = random.randint(0, 1280)
            y = random.randint(0, 960)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.circle(dashboard, (x, y), 5, color, -1)
            
    elif game_state.state == 'CRASH':
        overlay = dashboard.copy()
        cv2.rectangle(overlay, (0,0), (1280, 960), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.5, dashboard, 0.5, 0, dashboard)
        cv2.putText(dashboard, "CRASH", (500, 480), font, 3.0, (255, 255, 255), 5)
        
    return dashboard
