import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# Virtual camera intrinsics and distortion
f = 800.0
cx, cy = 320.0, 240.0
IMAGE_SIZE = (640, 480)
K = np.array([
    [f, 0, cx],
    [0, f, cy],
    [0, 0, 1]
], dtype=np.float64)

# k1, k2, p1, p2
DIST_COEFFS = np.array([-0.3, 0.1, 0.001, 0.001, 0.0], dtype=np.float64)

# Stereo rig parameters
B = 0.12 # Baseline in meters

def get_distort_maps(img_size, K, dist):
    """
    Computes maps to apply distortion to a clean image.
    We iterate over all target distorted pixel coordinates (u_d, v_d),
    undistort them to find the corresponding ideal/undistorted pixel 
    coordinates (u_u, v_u) to sample from the clean image.
    """
    v, u = np.mgrid[0:img_size[1], 0:img_size[0]]
    pts_d = np.stack([u.ravel(), v.ravel()], axis=-1).astype(np.float32)
    # Undistort the distorted coordinates to find where they map in the clean image
    pts_u = cv2.undistortPoints(pts_d, K, dist, P=K)
    map_x = pts_u[:, 0, 0].reshape((img_size[1], img_size[0])).astype(np.float32)
    map_y = pts_u[:, 0, 1].reshape((img_size[1], img_size[0])).astype(np.float32)
    return map_x, map_y

MAP_X, MAP_Y = get_distort_maps(IMAGE_SIZE, K, DIST_COEFFS)

def apply_distortion(img):
    """Apply lens distortion using reverse un-distortion mapping."""
    # We sample the clean source image at the undistorted coordinates map_x, map_y
    # to yield naturally distorted pixels in the output.
    distorted = cv2.remap(img, MAP_X, MAP_Y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return distorted

def add_noise(img, sigma=2.0):
    noise = np.random.normal(0, sigma, img.shape)
    img_noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img_noisy

def generate_panel_texture(size=300):
    """Creates a synthetic dataset of a solar-panel-like pattern."""
    panel = np.ones((size, size), dtype=np.uint8) * 255
    # Draw thicker ArUco-like grid
    grid_spacing = size // 5
    for i in range(1, 5):
        cv2.line(panel, (i * grid_spacing, 0), (i * grid_spacing, size), 0, 4)
        cv2.line(panel, (0, i * grid_spacing), (size, i * grid_spacing), 0, 4)
    # Draw random dots
    np.random.seed(42) # Consistent pattern
    for _ in range(80):
        x, y = np.random.randint(0, size, 2)
        r = np.random.randint(2, 6)
        cv2.circle(panel, (x, y), r, 50, -1)
    
    # Outer black box for contrast
    cv2.rectangle(panel, (0,0), (size-1, size-1), 0, 10)
    return panel

PANEL_TEXTURE = generate_panel_texture()

def generate_stereo_frame(pitch_deg, roll_deg, z_dist=2.5, add_noise_flag=True):
    """
    Generates left and right images for a given pitch and roll.
    pitch: rotation around X axis
    roll: rotation around Z axis (or Y axis depending on convention, let's use Y here so roll affects left-right tilt)
    Wait, aero convention:
    Pitch = X-axis
    Yaw = Y-axis
    Roll = Z-axis
    Let's make Roll around Z.
    """
    # 1. Define panel corners in 3D locally
    # Let panel width/height be 0.6 meters physical size
    pw = 0.6
    ph = 0.6
    # Panel corners in local frame (Centered at origin)
    corners_local = np.array([
        [-pw/2, -ph/2, 0],
        [ pw/2, -ph/2, 0],
        [ pw/2,  ph/2, 0],
        [-pw/2,  ph/2, 0]
    ], dtype=np.float32)
    
    # 2. Apply Rotation (Pitch: X, Roll: Y - actually roll in CV can be Z but let's standardise on pitch=X, roll=Y, yaw=Z, 
    # to make it clearly visible tilt. We'll use Rotation.from_euler)
    # The prompt says Pitch and Roll.
    # Pitch: tilts up/down (rotation about X)
    # Roll: tilts left/right (rotates normal in X-Z plane, which is rotation about Y axis)
    r = Rotation.from_euler('xyz', [-pitch_deg, roll_deg, 0], degrees=True)
    R_matrix = r.as_matrix()
    
    corners_rot = (R_matrix @ corners_local.T).T
    
    # 3. Translate to world coords. We put the panel between left and right cameras at depth z_dist
    tc = np.array([B/2.0, 0, z_dist])
    corners_3d = corners_rot + tc
    
    # 4. Project onto Left and Right cameras
    # Left Camera (at origin)
    rvec_l = np.zeros((3,1))
    tvec_l = np.zeros((3,1))
    # Standard mathematically exact projection from 3D to 2D
    # Our distortion is 0 for the projection, we apply reverse distortion manually later
    corners_2d_l, _ = cv2.projectPoints(corners_3d, rvec_l, tvec_l, K, None)
    corners_2d_l = corners_2d_l.reshape(4, 2)
    
    # Right Camera (at x=B relative to Left camera, so its position is +B, thus tvec inside cv2 is -B)
    rvec_r = np.zeros((3,1))
    tvec_r = np.array([[-B], [0], [0]]) # Right camera sees the world shifted left by B
    corners_2d_r, _ = cv2.projectPoints(corners_3d, rvec_r, tvec_r, K, None)
    corners_2d_r = corners_2d_r.reshape(4, 2)
    
    # 5. Warp texture to projected corners
    panel_h, panel_w = PANEL_TEXTURE.shape
    texture_corners = np.array([
        [0, 0],
        [panel_w-1, 0],
        [panel_w-1, panel_h-1],
        [0, panel_h-1]
    ], dtype=np.float32)
    
    img_l_clean = np.ones((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.uint8) * 200 # grey background
    img_r_clean = np.ones((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.uint8) * 200
    
    H_L, _ = cv2.findHomography(texture_corners, corners_2d_l)
    H_R, _ = cv2.findHomography(texture_corners, corners_2d_r)
    
    img_l_clean = cv2.warpPerspective(PANEL_TEXTURE, H_L, (IMAGE_SIZE[0], IMAGE_SIZE[1]), 
                                      dst=img_l_clean, borderMode=cv2.BORDER_TRANSPARENT)
    img_r_clean = cv2.warpPerspective(PANEL_TEXTURE, H_R, (IMAGE_SIZE[0], IMAGE_SIZE[1]), 
                                      dst=img_r_clean, borderMode=cv2.BORDER_TRANSPARENT)
                                      
    # 6. Apply Lens Distortion via reverse-mapping
    img_l_dist = apply_distortion(img_l_clean)
    img_r_dist = apply_distortion(img_r_clean)
    
    # 7. Add Noise
    if add_noise_flag:
        img_l_dist = add_noise(img_l_dist)
        img_r_dist = add_noise(img_r_dist)
        
    ground_truth = {
        'pitch': pitch_deg,
        'roll': roll_deg,
        'Z_panel': z_dist
    }
    
    return img_l_dist, img_r_dist, ground_truth

if __name__ == "__main__":
    l, r, gt = generate_stereo_frame(10, 20)
    cv2.imwrite("test_l.png", l)
    cv2.imwrite("test_r.png", r)
    print(f"Test generated. GT: {gt}")
