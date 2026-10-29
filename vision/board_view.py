"""Turns a camera photo into a flat, straight-down picture of just the chess board."""
import numpy as np


def load_board_calibration(path: str) -> dict:
    """Read where the board's corners are in the photo, saved by calibration/board_corners.py.

    Raises FileNotFoundError if the file is missing.
    """
    raise NotImplementedError


def top_down_view(photo: np.ndarray, board_calibration: dict) -> np.ndarray:
    """Return a square picture of only the playing area, seen from straight above, with a1 at the bottom-left."""
    raise NotImplementedError
