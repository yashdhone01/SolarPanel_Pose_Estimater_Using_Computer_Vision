def evaluate_confidence(inlier_count, valid_points, z_std, align_error):
    """
    Evaluates pipeline metrics to determine physical validity of the estimate.
    Returns: status (str), confidence (str)
    """
    status = ""
    confidence = "HIGH"
    
    # 1. Feature check
    if inlier_count < 10:
        status = "LOW FEATURE CONFIDENCE"
        confidence = "LOW"
    # 2. Minimum point cloud density check
    elif valid_points < 100:
        status = "NO RELIABLE ESTIMATE"
        confidence = "LOW"
    # 3. Depth Variance check (physical unlikelihood on planar surface)
    elif z_std > 0.5: 
        status = "DEPTH UNSTABLE"
        confidence = "LOW"
    # 4. Nominal Operation check
    else:
        if align_error < 5.0:
            status = "ALIGNED"
        else:
            status = "MISALIGNED"
            
    return status, confidence
