"""Takes photos with the overhead camera and removes the fisheye bend from them."""
import cv2
import numpy as np


def open_camera(camera_settings: dict) -> cv2.VideoCapture:
    """Open the camera at the configured size, with brightness and colour fixed so photos stay comparable.

    Raises RuntimeError if the camera can't be opened or refuses the size.
    """
    raise NotImplementedError


def take_photo(camera: cv2.VideoCapture, old_frames_to_skip: int) -> np.ndarray:
    """Return a fresh colour photo, after throwing away frames the camera had already buffered.

    Raises RuntimeError if the camera stops giving pictures.
    """
    raise NotImplementedError


def load_lens_calibration(path: str) -> dict:
    """Read the lens measurements saved by calibration/camera_intrinsics.py. Raises FileNotFoundError if missing."""
    raise NotImplementedError


def undistort(photo: np.ndarray, lens_calibration: dict) -> np.ndarray:
    """Straighten the fisheye bend so straight board lines look straight. Same size as the input.

    Raises ValueError if the photo size doesn't match the size the lens was measured at.
    """
    raise NotImplementedError
