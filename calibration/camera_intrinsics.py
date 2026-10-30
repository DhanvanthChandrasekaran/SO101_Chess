"""Measures the camera lens from checkerboard photos and saves it to calibration/data/camera_intrinsics.yaml.

Usage: python calibration/camera_intrinsics.py --images FOLDER [--inner-corners COLS ROWS] [--square-mm MM]
       [--output-folder DIR]. The checkerboard size and lens model default to config/vision.yaml.
No printer? python calibration/camera_intrinsics.py --make-checkerboard OUT.png --screen-px WIDTH HEIGHT
       writes a checkerboard picture to show full screen on a laptop or phone.
Taking the photos: python calibration/camera_intrinsics.py --capture FOLDER saves one automatically whenever the
       board is held still somewhere new. Move it around: centre, corners, edges, tilted, nearer and farther.
"""
import argparse
import glob
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Run as a script from the repo root, so make the project folders importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision import camera  # noqa: E402

# Fewer photos than this rarely pins down a fisheye lens; take more from different angles and image corners.
MINIMUM_PHOTOS = 10
# Searching the full 2592x1944 photo for the board is too slow to do on every frame, so the quick search
# runs on a copy shrunk to this width; only photos about to be saved are checked at full size.
QUICK_SEARCH_WIDTH_PX = 864


def make_checkerboard_picture(inner_corners: tuple[int, int], screen_width: int, screen_height: int) -> np.ndarray:
    """Return a black-and-white checkerboard picture exactly the size of the screen, as big as fits.

    A white border of at least one square is left around it, because the finder needs to see the board's edge.
    """
    columns, rows = inner_corners
    squares_across, squares_down = columns + 1, rows + 1
    square_px = min(screen_width // (squares_across + 2), screen_height // (squares_down + 2))
    picture = np.full((screen_height, screen_width), 255, np.uint8)
    left = (screen_width - squares_across * square_px) // 2
    top = (screen_height - squares_down * square_px) // 2
    for row in range(squares_down):
        for column in range(squares_across):
            if (row + column) % 2 == 0:
                picture[top + row * square_px:top + (row + 1) * square_px,
                        left + column * square_px:left + (column + 1) * square_px] = 0
    return picture


def find_checkerboard(grey: np.ndarray, inner_corners: tuple[int, int]) -> np.ndarray | None:
    """Return the inner-corner pixel positions found in one black-and-white photo, or None if the board isn't all visible."""
    found, corners = cv2.findChessboardCornersSB(grey, inner_corners,
                                                 cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE |
                                                 cv2.CALIB_CB_ACCURACY)
    return corners.reshape(-1, 1, 2).astype(np.float64) if found else None


def outline_distance(outline: np.ndarray, other_outline: np.ndarray) -> float:
    """Average distance in pixels between the four outer corners of two board sightings.

    The board can be seen the other way round (turned 180 degrees), so both corner orders are tried.
    """
    same_order = np.linalg.norm(outline - other_outline, axis=1).mean()
    turned_round = np.linalg.norm(outline - np.roll(other_outline, 2, axis=0), axis=1).mean()
    return min(same_order, turned_round)


def capture_checkerboard_photos(output_folder: str, inner_corners: tuple[int, int], capture_settings: dict) -> int:
    """Watch the camera and save a raw photo each time the board is held still somewhere new. Returns photos saved."""
    with open("config/hardware.yaml") as f:
        camera_settings = yaml.safe_load(f)["camera"]
    live_camera = camera.open_camera(camera_settings)
    os.makedirs(output_folder, exist_ok=True)
    columns = inner_corners[0]
    saved_outlines, previous_corners = [], None
    figure, axes = plt.subplots(figsize=(12, 9))
    plt.ion()
    try:
        while plt.fignum_exists(figure.number) and len(saved_outlines) < capture_settings["target_photos"]:
            photo = camera.take_photo(live_camera, 0)
            image_diagonal_px = float(np.hypot(photo.shape[0], photo.shape[1]))
            shrink = QUICK_SEARCH_WIDTH_PX / photo.shape[1]
            small = cv2.resize(photo, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
            found, small_corners = cv2.findChessboardCorners(
                cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), inner_corners,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)

            if not found:
                status, previous_corners = "board not found: show the whole board", None
            else:
                corners = small_corners.reshape(-1, 2) / shrink  # back to full-size pixels
                outline = corners[[0, columns - 1, -1, -columns]]  # the four outer inner-corners
                held_still = (previous_corners is not None and
                              np.linalg.norm(corners - previous_corners, axis=1).max() <= capture_settings["still_px"])
                far_enough = capture_settings["min_change_frac"] * image_diagonal_px
                somewhere_new = all(outline_distance(outline, saved) > far_enough for saved in saved_outlines)
                previous_corners = corners
                if not somewhere_new:
                    status = "already have this spot: move or tilt the board"
                elif not held_still:
                    status = "new spot: hold still ..."
                elif find_checkerboard(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), inner_corners) is None:
                    status = "full-size check failed (blur or glare?): hold still"
                else:
                    path = os.path.join(output_folder, f"{time.strftime('%Y%m%d_%H%M%S')}_{len(saved_outlines):02d}.jpg")
                    cv2.imwrite(path, photo, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_outlines.append(outline)
                    status = f"SAVED {os.path.basename(path)}"
                    print(status)

            preview = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            for saved in saved_outlines:  # where photos were already taken, so the gaps are easy to see
                cv2.polylines(preview, [np.round(saved * shrink).astype(np.int32)], True, (0, 200, 0), 2)
            if found:
                cv2.drawChessboardCorners(preview, inner_corners, small_corners, found)
            axes.clear()
            axes.imshow(preview)
            axes.axis("off")
            axes.set_title(f"{len(saved_outlines)}/{capture_settings['target_photos']} saved (green = done)   {status}")
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        live_camera.release()
    print(f"{len(saved_outlines)} photos in {output_folder}")
    return len(saved_outlines)


def main(command_line_args: list[str] | None = None) -> int:
    """Run the calibration described at the top of this file."""
    with open("config/vision.yaml") as f:
        vision_settings = yaml.safe_load(f)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", help="folder of checkerboard photos (.jpg or .png)")
    parser.add_argument("--capture", metavar="FOLDER", help="only take checkerboard photos automatically, into FOLDER")
    parser.add_argument("--make-checkerboard", metavar="OUT.png", help="only write a checkerboard picture for a screen")
    parser.add_argument("--screen-px", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="screen resolution in pixels, used with --make-checkerboard")
    parser.add_argument("--inner-corners", type=int, nargs=2, metavar=("COLS", "ROWS"),
                        default=vision_settings["checkerboard"]["inner_corners"])
    parser.add_argument("--square-mm", type=float, default=vision_settings["checkerboard"]["square_mm"])
    parser.add_argument("--output-folder", default="calibration/data")
    args = parser.parse_args(command_line_args)
    if None in args.inner_corners:
        parser.error("set checkerboard.inner_corners in config/vision.yaml or pass --inner-corners")

    if args.capture:
        capture_checkerboard_photos(args.capture, tuple(args.inner_corners), vision_settings["checkerboard"]["capture"])
        return 0

    if args.make_checkerboard:
        if args.screen_px is None:
            parser.error("--make-checkerboard needs --screen-px WIDTH HEIGHT")
        columns, rows = args.inner_corners
        cv2.imwrite(args.make_checkerboard, make_checkerboard_picture(tuple(args.inner_corners), *args.screen_px))
        print(f"Saved {args.make_checkerboard}: {columns + 1}x{rows + 1} squares ({columns}x{rows} inner corners).\n"
              "Show it full screen at 100% zoom, measure across all squares with a ruler, divide by the number of\n"
              "squares, and put the result in config/vision.yaml: checkerboard.square_mm.")
        return 0

    if args.images is None:
        parser.error("give --images FOLDER (or --capture FOLDER, or --make-checkerboard)")
    if args.square_mm is None:
        parser.error("checkerboard.square_mm is still TODO in config/vision.yaml; measure it or pass --square-mm")
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
        corners = find_checkerboard(cv2.imread(path, cv2.IMREAD_GRAYSCALE), inner_corners)
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
