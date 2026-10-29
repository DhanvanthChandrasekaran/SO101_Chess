"""Capture, locked exposure, undistort."""
import cv2
import numpy as np


def open_camera(camera_cfg: dict) -> cv2.VideoCapture:
    """Open the camera at the configured resolution with locked exposure/WB; raise RuntimeError on failure."""
    raise NotImplementedError


def grab(cap: cv2.VideoCapture, flush: int) -> np.ndarray:
    """Drop `flush` stale frames, return a fresh (H,W,3) uint8 BGR raw frame; raise RuntimeError on failure."""
    raise NotImplementedError


def load_intrinsics(path: str) -> dict:
    """Load {"K", "D", "model", "size"} from calibration/data/camera_intrinsics.yaml."""
    raise NotImplementedError


def undistort(img: np.ndarray, intr: dict) -> np.ndarray:
    """Raw frame -> undistorted frame of the same shape; raise ValueError on size mismatch."""
    raise NotImplementedError
