# 3D Solar Panel Pose Estimation via Stereo Vision

This project is a fully functional, zero-dependency (no external datasets or hardware required) synthetic Computer Vision pipeline. It simulates a stereo-vision environment to estimate the 3D pose (pitch and roll angles) of a solar panel relative to the sun.

The entire pipeline is built from the ground up using **explicit mathematical formulations** for stereo rectification, disparity unprojection, and Singular Value Decomposition (SVD) plane fitting.

## Features

- **Synthetic Environment**: Automatically generates a virtual stereo rig, projects an ArUco-styled solar panel at varied 3D orientations, and applies realistic lens distortion and Gaussian noise.
- **Stereo Calibration**: Synthesizes 25 distinct checkerboard poses to calibrate the synthetic cameras internally and externally, producing robust rectification maps (sub-pixel reprojection error `< 0.8px`).
- **Feature Matching**: Implements `cv2.SIFT` keypoint detection, brute-force KNN matching, Lowe's Ratio test, and RANSAC homography estimation.
- **Dense Depth Computation**: Uses StereoSGBM block matching on rectified images and explicitly unprojects disparity into 3D point clouds `Z = (f * B) / d`.
- **Plane Fitting & Pose Extraction**: Centers the 3D point cloud and utilizes `np.linalg.svd` to extract the normal vector and analytically compute Pitch and Roll constraints.
- **Interactive Event-Driven UI**: A custom 6-panel `matplotlib` dashboard displaying the pipeline stages and a live 3D visualization of the Panel vs Sun orientation.

## Prerequisites

- **Python 3.11+**
- `opencv-python` (cv2)
- `numpy`
- `matplotlib`
- `scipy`

## Usage

1. Clone the repository and navigate to the root directory.
2. Run the main execution script:
   ```bash
   python main.py
   ```
3. The script will briefly cache the calibration matrix to `calibration_data.npz`, and then boot the interactive 6-panel Matplotlib UI.

## Interactive Controls

The project has been architected as an event-driven system to allow clear cause-and-effect demonstration during evaluation.

- **`Arrow Keys`**: Move the Sun's simulated position across the sky. You will instantly see the orange Sun direction vector update on Panel 6.
- **`SPACE` Key**: Trigger the Computer Vision pipeline! This locks in the sun, randomizes the Ground Truth panel orientation, generates the synthetic images, computes the SIFT/Depth/SVD algorithms, and plots the final "Alignment Error" and visual maps.
- **`R` Key**: Resets the Sun's state to defaults.
- **`Q` Key**: Safely quits the interactive loop.

## Architecture

- `main.py`: The core event-driven loop and execution glue.
- `interactive_display.py`: The `matplotlib` grid and keyboard asynchronous event polling hooks.
- `simulate.py`: Virtual camera projection mapping and reverse-distortion synthesis.
- `calibrate.py`: Synthetic 3D checkerboard generation and cv2 rectification map dumping.
- `sift_match.py`: Feature isolation returning inlier keypoints and panel segmentation bounds.
- `depth.py`: Sub-pixel disparity block matching and explicit 3D space unprojection.
- `plane_fit.py`: The mathematical SVD core responsible for analytical pitch and roll extraction from the point cloud.
