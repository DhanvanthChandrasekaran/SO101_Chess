"""Measures the camera lens from checkerboard photos and saves it to calibration/data/camera_intrinsics.yaml.

Usage: python calibration/camera_intrinsics.py --images FOLDER [--inner-corners COLS ROWS] [--square-mm MM]
       [--output-folder DIR]. The checkerboard size and lens model default to config/vision.yaml.
"""
import argparse
import glob
import os

import cv2
import numpy as np
import yaml

# Fewer photos than this rarely pins down a fisheye lens; take more from different angles and image corners.
MINIMUM_PHOTOS = 10


def find_checkerboard(photo_path: str, inner_corners: tuple[int, int]) -> np.ndarray | None:
    """Return the inner-corner pixel positions found in one photo, or None if the whole board isn't visible."""
    grey = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
    found, corners = cv2.findChessboardCornersSB(grey, inner_corners,
                                                 cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE |
                                                 cv2.CALIB_CB_ACCURACY)
    return corners.reshape(-1, 1, 2).astype(np.float64) if found else None


def main(command_line_args: list[str] | None = None) -> int:
    """Run the calibration described at the top of this file."""
    with open("config/vision.yaml") as f:
        vision_settings = yaml.safe_load(f)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, help="folder of checkerboard photos (.jpg or .png)")
    parser.add_argument("--inner-corners", type=int, nargs=2, metavar=("COLS", "ROWS"),
                        default=vision_settings["checkerboard"]["inner_corners"])
    parser.add_argument("--square-mm", type=float, default=vision_settings["checkerboard"]["square_mm"])
    parser.add_argument("--output-folder", default="calibration/data")
    args = parser.parse_args(command_line_args)
    if None in args.inner_corners or args.square_mm is None:
        parser.error("checkerboard size is still TODO in config/vision.yaml; set it or pass --inner-corners/--square-mm")
    inner_corners = tuple(args.inner_corners)
    lens_model = vision_settings["lens_model"]

    # Where each inner corner sits on the printed board, in metres, with the board lying flat at z = 0.
    columns, rows = inner_corners
    corners_on_paper = np.zeros((columns * rows, 1, 3), np.float64)
    corners_on_paper[:, 0, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * args.square_mm / 1000

    photo_paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")) + glob.glob(os.path.join(args.images, "*.png")))
    corners_in_photos, image_size = [], None
    for path in photo_paths:
        height, width = cv2.imread(path, cv2.IMREAD_GRAYSCALE).shape
        if image_size not in (None, (width, height)):
            print(f"  skipped {os.path.basename(path)}: size {width}x{height} differs from the first photo")
            continue
        image_size = (width, height)
        corners = find_checkerboard(path, inner_corners)
        print(f"  {os.path.basename(path)}: {'found' if corners is not None else 'board NOT found'}")
        if corners is not None:
            corners_in_photos.append(corners)
    if len(corners_in_photos) < MINIMUM_PHOTOS:
        print(f"Only {len(corners_in_photos)} usable photos; need at least {MINIMUM_PHOTOS}.")
        return 1

    stop_when = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-9)
    if lens_model == "fisheye":
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        average_error_px, camera_matrix, distortion, _, _ = cv2.fisheye.calibrate(
            [corners_on_paper] * len(corners_in_photos), corners_in_photos, image_size, None, None,
            flags=flags, criteria=stop_when)
    else:
        average_error_px, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
            [corners_on_paper.astype(np.float32)] * len(corners_in_photos),
            [c.astype(np.float32) for c in corners_in_photos], image_size, None, None, criteria=stop_when)
    print(f"Used {len(corners_in_photos)} photos; average corner error {average_error_px:.3f} px "
          "(under ~0.5 px is good).")

    os.makedirs(args.output_folder, exist_ok=True)
    output_path = os.path.join(args.output_folder, "camera_intrinsics.yaml")
    with open(output_path, "w") as f:
        yaml.safe_dump({"camera_matrix": camera_matrix.tolist(), "distortion": distortion.reshape(-1).tolist(),
                        "lens_model": lens_model, "image_size": list(image_size)}, f, sort_keys=False)
    print(f"Saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
