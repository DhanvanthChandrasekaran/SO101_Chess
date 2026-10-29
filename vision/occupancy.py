"""Top-down board image -> 8x8 occupied grid."""
import numpy as np


def square_scores(top: np.ndarray, empty_top: np.ndarray, vision_cfg: dict) -> np.ndarray:
    """Per-square difference vs the empty-board reference -> (8,8) float32 [rank, file]."""
    raise NotImplementedError


def occupancy(scores: np.ndarray, vision_cfg: dict) -> np.ndarray:
    """Threshold scores -> (8,8) bool [rank, file]."""
    raise NotImplementedError
