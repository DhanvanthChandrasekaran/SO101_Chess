"""Takes photos with the overhead camera and removes the fisheye bend from them."""
import subprocess

import cv2
import numpy as np
import yaml

# Building the undistort lookup table takes a moment at full resolution, so it is built once per lens and reused.
_undistort_tables = {}


def open_camera(camera_settings: dict) -> cv2.VideoCapture:
    """Open the camera at the configured size, with brightness and colour fixed so photos stay comparable.

    Raises RuntimeError if the camera can't be opened or refuses the size.
    """
    # Comparing against the empty-board photo only works if exposure never drifts, so refuse to run on auto.
    if camera_settings.get("exposure") is None or camera_settings.get("white_balance") is None:
        raise RuntimeError("Set camera exposure and white_balance in config/hardware.yaml before opening the camera.")
    device = camera_settings["device"]
    camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {device} (is another app like Cheese using it?)")
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*camera_settings["fourcc"]))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_settings["width"])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_settings["height"])
    size_given = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if size_given != (camera_settings["width"], camera_settings["height"]):
        camera.release()
        raise RuntimeError(f"Camera gave {size_given[0]}x{size_given[1]}, "
                           f"not the {camera_settings['width']}x{camera_settings['height']} in config.")

    # The automatic modes must be switched off before the fixed values are accepted, hence two calls.
    autos_off = "auto_exposure=1,white_balance_automatic=0"
    fixed_values = (f"exposure_time_absolute={camera_settings['exposure']},"
                    f"white_balance_temperature={camera_settings['white_balance']}")
    for controls in (autos_off, fixed_values):
        result = subprocess.run(["v4l2-ctl", "-d", device, "-c", controls], capture_output=True, text=True)
        if result.returncode != 0:
            camera.release()
            raise RuntimeError(f"v4l2-ctl could not set {controls}: {result.stderr.strip()}")
    return camera


def take_photo(camera: cv2.VideoCapture, old_frames_to_skip: int) -> np.ndarray:
    """Return a fresh colour photo, after throwing away frames the camera had already buffered.

    Raises RuntimeError if the camera stops giving pictures.
    """
    for _ in range(old_frames_to_skip):
        camera.grab()
    worked, photo = camera.read()
    if not worked or photo is None:
        raise RuntimeError("The camera did not return a photo.")
    return photo


def load_lens_calibration(path: str) -> dict:
    """Read the lens measurements saved by calibration/camera_intrinsics.py. Raises FileNotFoundError if missing."""
    with open(path) as f:
        saved = yaml.safe_load(f)
    return {
        "camera_matrix": np.array(saved["camera_matrix"], dtype=np.float64),
        "distortion": np.array(saved["distortion"], dtype=np.float64),
        "lens_model": saved["lens_model"],
        "image_size": tuple(int(n) for n in saved["image_size"]),
    }


def undistort(photo: np.ndarray, lens_calibration: dict) -> np.ndarray:
    """Straighten the fisheye bend so straight board lines look straight. Same size as the input.

    Raises ValueError if the photo size doesn't match the size the lens was measured at.
    """
    width_and_height = (photo.shape[1], photo.shape[0])
    if width_and_height != tuple(lens_calibration["image_size"]):
        raise ValueError(f"Photo is {width_and_height[0]}x{width_and_height[1]} but the lens was measured at "
                         f"{lens_calibration['image_size'][0]}x{lens_calibration['image_size'][1]}.")
    columns, rows = undistort_lookup_table(lens_calibration)
    return cv2.remap(photo, columns, rows, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def undistort_lookup_table(lens_calibration: dict) -> tuple[np.ndarray, np.ndarray]:
    """For every pixel of the straightened photo, where to read it from in the raw photo (built once per lens)."""
    camera_matrix, distortion = lens_calibration["camera_matrix"], lens_calibration["distortion"]
    size = tuple(lens_calibration["image_size"])
    key = (lens_calibration["lens_model"], camera_matrix.tobytes(), distortion.tobytes(), size)
    if key not in _undistort_tables:
        no_rotation = np.eye(3)
        if lens_calibration["lens_model"] == "fisheye":
            # balance 0 zooms in until every output pixel comes from inside the raw photo.
            straight_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                camera_matrix, distortion, size, no_rotation, balance=0)
            table = cv2.fisheye.initUndistortRectifyMap(camera_matrix, distortion, no_rotation, straight_matrix,
                                                        size, cv2.CV_16SC2)
        else:
            straight_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion, size, alpha=0)
            table = cv2.initUndistortRectifyMap(camera_matrix, distortion, no_rotation, straight_matrix,
                                                size, cv2.CV_16SC2)
        _undistort_tables[key] = table
    return _undistort_tables[key]
