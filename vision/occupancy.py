"""Decides which of the 64 squares have a piece on them, by comparing with a photo of the empty board."""
import numpy as np


def square_difference_scores(top_down: np.ndarray, empty_board_top_down: np.ndarray,
                             vision_settings: dict) -> np.ndarray:
    """Return an 8x8 grid of numbers: how different each square's centre looks from the empty board.

    Bigger means more likely to hold a piece. Row 0 is rank 1. Raises ValueError if the two pictures differ in size.
    """
    raise NotImplementedError


def occupied_squares(difference_scores: np.ndarray, vision_settings: dict) -> np.ndarray:
    """Return an 8x8 grid of True/False: True where the score is above the threshold in the settings."""
    raise NotImplementedError
