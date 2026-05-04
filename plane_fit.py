import numpy as np

def fit_plane(points_3d):
    """
    Fits a plane to a 3D point cloud using SVD.
    Returns: normal, pitch, roll, mean_residual
    """
    # 1. Check valid point count
    if len(points_3d) < 3:
        return None, 0.0, 0.0, 0.0
        
    # 2. Subtract centroid
    centroid = np.mean(points_3d, axis=0)
    centered_points = points_3d - centroid
    
    # 3. SVD for plane fitting
    # U, S, Vt = np.linalg.svd(centered_points, full_matrices=False)
    # The normal vector is the last row of Vt (corresponds to smallest singular value)
    U, S, Vt = np.linalg.svd(centered_points, full_matrices=False)
    n = Vt[2, :]
    
    # Ensuring normal points such that n[2] is positive, 
    # to make arctan2(y, x) near 0 when fronto-parallel.
    if n[2] < 0:
        n = -n
        
    # 4. Compute Pitch and Roll
    # Explicit math as requested:
    pitch = np.degrees(np.arctan2(n[1], n[2]))
    roll = np.degrees(np.arctan2(n[0], n[2]))
    
    # Calculate fit residual (mean absolute distance to plane)
    # Distance of point P to plane = |(P - centroid) dot n|
    distances = np.abs(np.dot(centered_points, n))
    mean_residual = np.mean(distances)
    
    return n, pitch, roll, mean_residual, centroid
