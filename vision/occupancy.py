"""Decides which of the 64 squares have a piece on them, by comparing with a photo of the empty board."""
import numpy as np


def square_difference_scores(top_down: np.ndarray, empty_board_top_down: np.ndarray,
                             vision_settings: dict) -> np.ndarray:
    """Return an 8x8 grid of numbers: how different each square's centre looks from the empty board.

    Bigger means more likely to hold a piece. Row 0 is rank 1. Raises ValueError if the two pictures differ in size.
    """
    if top_down.shape != empty_board_top_down.shape:
        raise ValueError(f"Board picture is {top_down.shape} but the empty-board picture is {empty_board_top_down.shape}.")
    # Pieces are the same browns and tans as the squares, so instead of looking for a piece colour we look for
    # any change from the empty board: colour distance per pixel, averaged over a patch.
    colour_change = np.linalg.norm(top_down.astype(np.float32) - empty_board_top_down.astype(np.float32), axis=2)

    square_px = top_down.shape[0] / 8
    half_patch_px = vision_settings["patch_frac"] * square_px / 2
    scores = np.zeros((8, 8), np.float32)
    for rank_index in range(8):
        for file_index in range(8):
            # Only the centre patch is used: tall pieces lean outward in the photo and can spill onto neighbours.
            centre_u = (file_index + 0.5) * square_px
            centre_v = (7 - rank_index + 0.5) * square_px  # rank 1 is at the bottom of the picture
            patch = colour_change[round(centre_v - half_patch_px):round(centre_v + half_patch_px),
                                  round(centre_u - half_patch_px):round(centre_u + half_patch_px)]
            scores[rank_index, file_index] = patch.mean()
    return scores


def occupied_squares(difference_scores: np.ndarray, vision_settings: dict) -> np.ndarray:
    """Return an 8x8 grid of True/False: True where the score is above the threshold in the settings.

    Raises ValueError while the threshold is still unset in config/vision.yaml.
    """
    if vision_settings.get("threshold") is None:
        raise ValueError("Set threshold in config/vision.yaml (tune it with tools/vision_live.py first).")
    return difference_scores > vision_settings["threshold"]
