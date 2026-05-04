# 3D Solar Panel Pose Estimation via Synthetic Stereo Vision
## Project Explanation & Technical Justification

### What This Project Does
This project implements a complete, closed-loop, event-driven computer vision system that mathematically estimates the 3D orientation (pitch and roll) of a solar panel purely from stereoscopic camera feeds. It features a custom synthetic physics engine that generates realistic camera distortion, renders virtual stereoscopic images of an ArUco-styled solar panel tracking a moving sun, and uses raw mathematics to back-calculate the panel's alignment error relative to the sun.

### Why It Is Effective & Useful
In space exploration and defense contexts, hardware simplicity and sensor redundancy are paramount. 
1. **Passive Sensing**: Traditional tracking methods often rely on active lidar or radar which emit detectable signatures and require high power. This system uses entirely passive optical stereoscopy, meaning it emits zero signals.
2. **GPS / IMU Independence**: In GPS-denied environments or situations where onboard IMUs suffer from drift or mechanical failure, visual stereoscopy provides an absolute, drift-free orientation measurement relative to the camera rig.
3. **Closed-Loop Verification**: It allows a central control system to visually verify if a mechanical servo actually reached the requested sun-tracking orientation, rather than blindly trusting the servo's internal encoder.

### Tech Stack Justification
**Python, OpenCV, NumPy, SciPy**
The stack was chosen to prioritize mathematical explicitness, deterministic execution, and rapid development.
- **Why not Deep Learning?** Neural networks are exceptionally powerful but suffer from "hallucinations" and lack mathematical proofs for their outputs. In mission-critical alignment tasks, a purely geometric pipeline (using epipolar geometry and projective math) guarantees predictable bounds of error and is easier to certify for safety standards.
- **Why OpenCV/NumPy?** They provide lightning-fast, highly optimized C++ backends for matrix operations while keeping the frontend code expressive and readable.

### Algorithm Justifications
1. **Camera Calibration (`cv2.calibrateCamera` & `cv2.stereoCalibrate`)**:
   Instead of assuming perfect cameras, the pipeline simulates realistic lens distortion ($k_1, k_2, p_1, p_2$) and calibrates them using 25 synthetic checkerboard poses. This proves the pipeline can handle off-the-shelf, imperfect optical hardware.

2. **Feature Extraction (`cv2.SIFT` & Lowe's Ratio Test)**:
   Scale-Invariant Feature Transform (SIFT) was chosen over simpler algorithms (like ORB/FAST) because it is highly robust against affine transformations and harsh directional lighting (which is expected in solar tracking). Lowe's Ratio test efficiently filters out ambiguous matches common in repetitive structural textures.

3. **Dense Depth Mapping (StereoSGBM)**:
   Semi-Global Block Matching is used instead of basic Block Matching. SGBM enforces smoothness constraints across multiple 1D paths across the image, which drastically reduces noise on flat, texture-sparse surfaces like solar panels, resulting in a cleaner unprojected point cloud.

4. **Planar Pose Extraction (Singular Value Decomposition - SVD)**:
   Once the 3D point cloud of the panel is unprojected from the disparity map ($Z = \frac{f \cdot B}{d}$), we center the cloud and apply SVD (`np.linalg.svd`). The normal vector to the plane is mathematically guaranteed to be the singular vector corresponding to the smallest singular value. This is the most robust, deterministic method for extracting a plane from a noisy point cloud without relying on iterative sampling methods like RANSAC, guaranteeing a stable $O(n)$ time complexity execution per frame.
