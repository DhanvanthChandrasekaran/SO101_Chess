"""You click the four board corners (a1, h1, h8, a8) on a photo of the empty board; saves board.yaml and empty_board.png.

Usage: python calibration/board_corners.py [--photo RAW_PHOTO] [--lens LENS_YAML] [--no-lens]
       [--corners U V U V U V U V | --keep-corners] [--output-folder DIR]
After changing exposure, white balance or lighting (board and camera not moved): --keep-corners retakes only
the empty-board photo, using the corners already saved.
Click the OUTER corners of the 8x8 playing area (not the wooden border). Ignore the printed letters:
a1 is the dark corner square nearest the robot, on the robot's left.
"""
import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Run as a script from the repo root, so make the project folders importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.vision_live import draw_board_grid  # noqa: E402
from vision import board_view, camera  # noqa: E402

CORNER_NAMES = ["a1", "h1", "h8", "a8"]
ZOOM_HALF_WIDTH_PX = 60  # how much photo is shown around each rough click when placing it exactly


def click_corners(straight_photo: np.ndarray) -> np.ndarray | None:
    """Let the user click a1, h1, h8, a8 roughly, then once more on a zoomed view of each. None if cancelled."""
    figure, axes = plt.subplots(figsize=(14, 10))
    axes.imshow(cv2.cvtColor(straight_photo, cv2.COLOR_BGR2RGB))
    axes.set_title("Click roughly on the playing-area corners in this order: a1, h1, h8, a8\n"
                   "(a1 = dark corner nearest the robot, on its left; right-click undoes)")
    rough_clicks = plt.ginput(4, timeout=0)
    if len(rough_clicks) < 4:
        plt.close(figure)
        return None

    exact_clicks = []
    for name, (u, v) in zip(CORNER_NAMES, rough_clicks):
        axes.set_xlim(u - ZOOM_HALF_WIDTH_PX, u + ZOOM_HALF_WIDTH_PX)
        axes.set_ylim(v + ZOOM_HALF_WIDTH_PX, v - ZOOM_HALF_WIDTH_PX)
        axes.set_title(f"Click exactly on the {name} corner of the playing area")
        figure.canvas.draw()
        click = plt.ginput(1, timeout=0)
        exact_clicks.append(click[0] if click else (u, v))
    plt.close(figure)
    return np.array(exact_clicks, dtype=np.float64)


def photo_to_top_down_from(corner_pixels: np.ndarray, pixels_per_square: int) -> np.ndarray:
    """The 3x3 mapping that sends the clicked corners to the corners of the top-down picture (a1 bottom-left)."""
    side_px = 8 * pixels_per_square
    top_down_corners = np.array([[0, side_px], [side_px, side_px], [side_px, 0], [0, 0]], dtype=np.float32)
    return cv2.getPerspectiveTransform(corner_pixels.astype(np.float32), top_down_corners).astype(np.float64)


def main(command_line_args: list[str] | None = None) -> int:
    """Run the calibration described at the top of this file."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photo", help="raw photo of the empty board (default: take one with the camera)")
    parser.add_argument("--lens", default="calibration/data/camera_intrinsics.yaml")
    parser.add_argument("--no-lens", action="store_true", help="use the photo as it is (trial only)")
    parser.add_argument("--corners", type=float, nargs=8, metavar="U_V",
                        help="a1, h1, h8, a8 positions in the straightened photo, instead of clicking")
    parser.add_argument("--keep-corners", action="store_true",
                        help="reuse the corners in the existing board.yaml; only retake the empty-board photo")
    parser.add_argument("--output-folder", default="calibration/data")
    args = parser.parse_args(command_line_args)
    if args.keep_corners and args.corners:
        parser.error("use either --keep-corners or --corners, not both")
    # A calibration made on a bent photo would not fit the straightened photos the game uses.
    if args.no_lens and os.path.realpath(args.output_folder) == os.path.realpath("calibration/data"):
        parser.error("--no-lens results must not go to calibration/data; pass --output-folder (e.g. data/images/trial)")
    with open("config/vision.yaml") as f:
        pixels_per_square = int(yaml.safe_load(f)["px_per_square"])

    if args.photo:
        photo = cv2.imread(args.photo)
        if photo is None:
            parser.error(f"could not read {args.photo}")
    else:
        with open("config/hardware.yaml") as f:
            camera_settings = yaml.safe_load(f)["camera"]
        live_camera = camera.open_camera(camera_settings)
        photo = camera.take_photo(live_camera, camera_settings["flush_frames"])
        live_camera.release()
    straight_photo = photo if args.no_lens else camera.undistort(photo, camera.load_lens_calibration(args.lens))

    if args.corners:
        corner_pixels = np.array(args.corners, dtype=np.float64).reshape(4, 2)
    elif args.keep_corners:
        # Only the light or exposure changed, not where the board sits, so the old clicks are still right.
        existing_board_path = os.path.join(args.output_folder, "board.yaml")
        if not os.path.exists(existing_board_path):
            parser.error(f"{existing_board_path} is missing: run once without --keep-corners to click the corners")
        corner_pixels = board_view.load_board_calibration(existing_board_path)["corner_pixels"]
    else:
        corner_pixels = click_corners(straight_photo)
        if corner_pixels is None:
            print("Cancelled; nothing saved.")
            return 1
    photo_to_top_down = photo_to_top_down_from(corner_pixels, pixels_per_square)
    board_calibration = {"photo_to_top_down": photo_to_top_down, "pixels_per_square": pixels_per_square,
                         "corner_pixels": corner_pixels}
    top_down = board_view.top_down_view(straight_photo, board_calibration)

    if not args.corners:  # show the result before saving, so a wrong click order is caught here
        figure, axes = plt.subplots(figsize=(8, 8))
        axes.imshow(cv2.cvtColor(draw_board_grid(top_down), cv2.COLOR_BGR2RGB))
        axes.set_title("Grid should sit on the square edges, a1 (dark) bottom-left.\n"
                       "Close this window, then answer in the terminal.")
        plt.show()
        if input("Save? [y/N] ").strip().lower() != "y":
            print("Not saved.")
            return 1

    os.makedirs(args.output_folder, exist_ok=True)
    with open(os.path.join(args.output_folder, "board.yaml"), "w") as f:
        yaml.safe_dump({"photo_to_top_down": photo_to_top_down.tolist(), "pixels_per_square": pixels_per_square,
                        "corner_pixels": corner_pixels.tolist()}, f, sort_keys=False)
    cv2.imwrite(os.path.join(args.output_folder, "empty_board.png"), top_down)
    print(f"Saved board.yaml and empty_board.png in {args.output_folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
