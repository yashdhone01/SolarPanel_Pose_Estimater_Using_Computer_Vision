import cv2
import numpy as np

def match_and_mask(img_l_rect, img_r_rect):
    """
    Detects SIFT features, matches them, and computes Homography.
    """
    # 1. Detect SIFT
    sift = cv2.SIFT_create(nfeatures=500)
    kp_l, des_l = sift.detectAndCompute(img_l_rect, None)
    kp_r, des_r = sift.detectAndCompute(img_r_rect, None)
    
    if des_l is None or des_r is None or len(kp_l) < 10 or len(kp_r) < 10:
        return None, None, None, None
        
    # 2. BFMatcher and KNN
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_l, des_r, k=2)
    
    # 3. Lowe's Ratio Test
    good_matches = []
    ratio_threshold = 0.75
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
            
    if len(good_matches) < 10:
        return None, None, None, None
        
    # 4. RANSAC Homography
    src_pts = np.float32([kp_l[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_r[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    
    if H is None:
        return None, None, None, None
        
    matchesMask = mask.ravel().tolist()
    
    # Visualization image
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)
    match_img = cv2.drawMatches(img_l_rect, kp_l, img_r_rect, kp_r, good_matches, None, **draw_params)
    
    # 5. Panel Mask (Convex Hull of inlier keypoints in Left image)
    panel_mask = np.zeros_like(img_l_rect)
    inlier_pts = []
    for i, m in enumerate(matchesMask):
        if m:
            inlier_pts.append(kp_l[good_matches[i].queryIdx].pt)
    
    if len(inlier_pts) > 3:
        inlier_pts = np.array(inlier_pts, dtype=np.int32)
        hull = cv2.convexHull(inlier_pts)
        cv2.fillConvexPoly(panel_mask, hull, 255)
    
    inlier_keypoints = (kp_l, kp_r, good_matches, matchesMask)
    return inlier_keypoints, match_img, H, panel_mask
