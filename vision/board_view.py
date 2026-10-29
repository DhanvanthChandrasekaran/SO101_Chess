"""Homography: undistorted image -> top-down board image."""
import numpy as np


def load_board_calib(path: str) -> dict:
    """Load {"H", "px_per_square", "corners_px"} from calibration/data/board.yaml."""
    raise NotImplementedError


def warp(img: np.ndarray, bcal: dict) -> np.ndarray:
    """Undistorted (H,W,3) uint8 -> top-down (N,N,3) uint8, a1 bottom-left."""
    raise NotImplementedError
