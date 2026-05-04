import cv2
import numpy as np

def compute_depth_and_pointcloud(img_l_dist, img_r_dist, calib_data, panel_mask):
    """
    Rectify, SGBM, Unproject 3D explicitly.
    """
    # 1. Rectify images
    map1_L = calib_data['map1_L']
    map2_L = calib_data['map2_L']
    map1_R = calib_data['map1_R']
    map2_R = calib_data['map2_R']
    
    img_l_rect = cv2.remap(img_l_dist, map1_L, map2_L, cv2.INTER_LINEAR)
    img_r_rect = cv2.remap(img_r_dist, map1_R, map2_R, cv2.INTER_LINEAR)
    
    # 2. SGBM Disparity
    # numDisparities=64, blockSize=11, P1=8*3*11**2, P2=32*3*11**2
    block_size = 11
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    
    # SGBM returns disparity multiplied by 16
    disparity_16 = sgbm.compute(img_l_rect, img_r_rect)
    disparity = disparity_16.astype(np.float32) / 16.0
    
    # 3. Disparity to Depth (Explicit Math)
    # Get parameters from rectified camera matrices or original
    P1 = calib_data['P1']
    f = P1[0, 0]
    cx = P1[0, 2]
    cy = P1[1, 2]
    B = calib_data['B']
    
    # Handle divide-by-zero safely
    Z = np.zeros_like(disparity)
    valid_disp = disparity > 0
    
    # Z = (f * B) / d
    Z[valid_disp] = (f * float(B)) / disparity[valid_disp]
    
    # 4. Unproject to 3D (Explicit Math)
    h, w = disparity.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    
    X = np.zeros_like(disparity)
    Y = np.zeros_like(disparity)
    
    # X = (u - cx) * Z / f
    X[valid_disp] = (u[valid_disp] - cx) * Z[valid_disp] / f
    
    # Y = (v - cy) * Z / f
    Y[valid_disp] = (v[valid_disp] - cy) * Z[valid_disp] / f
    
    # Stack to N x 3
    points_3d = np.stack([X, Y, Z], axis=-1)
    
    # 5. Filter to panel region + valid disparities
    # Erode the mask to avoid SGBM edge artifacts
    kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(panel_mask, kernel, iterations=2)
    
    mask_combined = (eroded_mask > 0) & valid_disp
    
    # Remove extremely far / erroneous points
    # Z=0 means invalid. Z>10 is probably wrong if panel is at ~2.5
    valid_Z = (Z > 0.5) & (Z < 10.0)
    mask_combined = mask_combined & valid_Z
    
    point_cloud = points_3d[mask_combined]
    
    # Create disparity map for visualization (normalized + JET + masked)
    disp_vis = np.zeros_like(disparity, dtype=np.uint8)
    y_idx, x_idx = np.where(mask_combined)
    
    if len(x_idx) > 0:
        d_min = disparity[mask_combined].min()
        d_max = disparity[mask_combined].max()
        # Scale manually to avoid normalizing background noise
        d_scaled = np.clip((disparity - d_min) / (d_max - d_min + 1e-6) * 255.0, 0, 255).astype(np.uint8)
        disp_vis[mask_combined] = d_scaled[mask_combined]
        
        # Calculate Depth at Centroid
        cy_mask = int(np.mean(y_idx))
        cx_mask = int(np.mean(x_idx))
        center_depth = Z[cy_mask, cx_mask]
        center_pt = (cx_mask, cy_mask)
        z_std = np.std(Z[mask_combined])
    else:
        center_depth = 0.0
        center_pt = (0, 0)
        z_std = 0.0
        
    disp_vis_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    disp_vis_color[~mask_combined] = [0, 0, 0]
    
    return img_l_rect, img_r_rect, disp_vis_color, point_cloud, center_depth, center_pt, len(x_idx), z_std
