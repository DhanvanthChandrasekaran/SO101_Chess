"""Shows the camera view with each square marked occupied or empty, for checking the vision by eye.

Live camera:  python tools/vision_live.py [--save DIR]           (press s to save the raw photo to DIR, close to quit)
Saved photos: python tools/vision_live.py --images DIR [--save OUT] (one window per photo, or files in OUT)
Other options: --calibration-folder DIR (default calibration/data), --no-lens (don't undistort).
Drag the threshold slider until every square with a piece has a green ring and no empty square does;
"Save threshold" writes it into config/vision.yaml.
"""
import argparse
import glob
import os
import re
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider

# Run as a script from the repo root, so make the project folders importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision import board_view, camera, occupancy  # noqa: E402

FILE_LETTERS = "abcdefgh"
VISION_CONFIG = "config/vision.yaml"
SLIDER_MAX_SCORE = 150  # highest threshold the slider offers; scores of occupied squares can go higher


def draw_board_grid(top_down: np.ndarray, occupied: np.ndarray | None = None) -> np.ndarray:
    """Return a copy of the top-down picture with the 8x8 grid, square names, and a green ring on occupied squares."""
    picture = top_down.copy()
    square_px = picture.shape[0] / 8
    for line in range(9):
        position = round(line * square_px)
        cv2.line(picture, (position, 0), (position, picture.shape[0]), (0, 0, 255), 1)
        cv2.line(picture, (0, position), (picture.shape[1], position), (0, 0, 255), 1)
    for rank_index in range(8):
        for file_index in range(8):
            left, top = round(file_index * square_px), round((7 - rank_index) * square_px)
            name = f"{FILE_LETTERS[file_index]}{rank_index + 1}"
            cv2.putText(picture, name, (left + 2, top + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
            if occupied is not None and occupied[rank_index, file_index]:
                centre = (round(left + square_px / 2), round(top + square_px / 2))
                cv2.circle(picture, centre, round(square_px * 0.3), (0, 255, 0), 2)
    return picture


def look_at_board(photo, lens_calibration, board_calibration, empty_board_top_down, vision_settings):
    """Run the vision steps on one raw photo. Returns (straight photo, top-down, scores); missing steps are None."""
    straight_photo = photo if lens_calibration is None else camera.undistort(photo, lens_calibration)
    if board_calibration is None:
        return straight_photo, None, None
    top_down = board_view.top_down_view(straight_photo, board_calibration)
    if empty_board_top_down is None:
        return straight_photo, top_down, None
    scores = occupancy.square_difference_scores(top_down, empty_board_top_down, vision_settings)
    return straight_photo, top_down, scores


def draw_panels(axes, straight_photo, top_down, scores, threshold, board_calibration) -> None:
    """Fill the three panels: photo with the board outline, top-down view with the grid, score heatmap."""
    for axis in axes:
        axis.clear()
        axis.axis("off")
    occupied = None
    if scores is not None and threshold is not None:
        occupied = occupancy.occupied_squares(scores, {"threshold": threshold})

    outlined = straight_photo.copy()
    if board_calibration is not None:
        corners = board_calibration["corner_pixels"].round().astype(np.int32)
        cv2.polylines(outlined, [corners], True, (0, 0, 255), 3)
    axes[0].imshow(cv2.cvtColor(outlined, cv2.COLOR_BGR2RGB))
    axes[0].set_title("photo, red = board outline")

    if top_down is not None:
        axes[1].imshow(cv2.cvtColor(draw_board_grid(top_down, occupied), cv2.COLOR_BGR2RGB))
        axes[1].set_title("top-down, a1 bottom-left, green = occupied")
    if scores is not None:
        axes[2].axis("on")
        rank_8_on_top = np.flipud(scores)
        axes[2].imshow(rank_8_on_top, cmap="magma", vmin=0, vmax=SLIDER_MAX_SCORE)
        for row in range(8):
            for column in range(8):
                score = rank_8_on_top[row, column]
                axes[2].text(column, row, f"{score:.0f}", ha="center", va="center", color="cyan", fontsize=7)
                if threshold is not None and score > threshold:
                    axes[2].add_patch(Rectangle((column - 0.45, row - 0.45), 0.9, 0.9, fill=False,
                                                edgecolor="lime", linewidth=2))
        axes[2].set_xticks(range(8), list(FILE_LETTERS))
        axes[2].set_yticks(range(8), [str(rank) for rank in range(8, 0, -1)])
        axes[2].set_title("difference scores, green box = above threshold")


def save_threshold(threshold: float) -> None:
    """Replace only the threshold value in config/vision.yaml, keeping everything else."""
    with open(VISION_CONFIG) as f:
        text = f.read()
    text, count = re.subn(r"^threshold:.*$", f"threshold: {threshold:.1f}  # set with the tools/vision_live.py slider",
                          text, flags=re.MULTILINE)
    if count != 1:
        print(f"Not saved: expected one 'threshold:' line in {VISION_CONFIG}, found {count}")
        return
    with open(VISION_CONFIG, "w") as f:
        f.write(text)
    print(f"Saved threshold: {threshold:.1f} in {VISION_CONFIG}")


def make_window(starting_threshold: float):
    """Open the window: three panels on top, the threshold slider and Save button underneath."""
    figure = plt.figure(figsize=(18, 7))
    axes = [figure.add_axes([0.01 + 0.33 * i, 0.15, 0.31, 0.78]) for i in range(3)]
    slider = Slider(figure.add_axes([0.15, 0.06, 0.55, 0.03]), "threshold", 0, SLIDER_MAX_SCORE,
                    valinit=starting_threshold)
    save_button = Button(figure.add_axes([0.78, 0.04, 0.12, 0.06]), "Save threshold")
    save_button.on_clicked(lambda _: save_threshold(slider.val))
    return figure, axes, slider, save_button


def main(command_line_args: list[str] | None = None) -> int:
    """Run the viewer described at the top of this file."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", help="folder of saved raw photos (.jpg or .png) instead of the live camera")
    parser.add_argument("--calibration-folder", default="calibration/data")
    parser.add_argument("--no-lens", action="store_true", help="use photos as they are, without undistorting")
    parser.add_argument("--save", metavar="DIR", help="with --images: write pictures here; live: s saves the raw photo here")
    args = parser.parse_args(command_line_args)
    with open(VISION_CONFIG) as f:
        vision_settings = yaml.safe_load(f)
    saved_threshold = vision_settings.get("threshold")
    starting_threshold = saved_threshold if saved_threshold is not None else SLIDER_MAX_SCORE / 2

    lens_path = os.path.join(args.calibration_folder, "camera_intrinsics.yaml")
    board_path = os.path.join(args.calibration_folder, "board.yaml")
    empty_board_path = os.path.join(args.calibration_folder, "empty_board.png")
    if not args.no_lens and not os.path.exists(lens_path):
        parser.error(f"{lens_path} is missing: run calibration/camera_intrinsics.py first, or pass --no-lens")
    lens_calibration = None if args.no_lens else camera.load_lens_calibration(lens_path)
    board_calibration = board_view.load_board_calibration(board_path) if os.path.exists(board_path) else None
    empty_board_top_down = cv2.imread(empty_board_path) if os.path.exists(empty_board_path) else None
    calibration = (lens_calibration, board_calibration, empty_board_top_down, vision_settings)

    if args.images:
        photo_paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")) + glob.glob(os.path.join(args.images, "*.png")))
        for path in photo_paths:
            straight_photo, top_down, scores = look_at_board(cv2.imread(path), *calibration)
            name = os.path.splitext(os.path.basename(path))[0]
            if args.save:  # no window: write the pictures, using the saved threshold
                figure = plt.figure(figsize=(18, 6))
                axes = [figure.add_subplot(1, 3, i + 1) for i in range(3)]
                draw_panels(axes, straight_photo, top_down, scores, saved_threshold, board_calibration)
                figure.suptitle(name)
                os.makedirs(args.save, exist_ok=True)
                cv2.imwrite(os.path.join(args.save, f"{name}_straight.jpg"), straight_photo)
                figure.savefig(os.path.join(args.save, f"{name}_view.png"), dpi=100)
                plt.close(figure)
                print(f"saved {name}")
                continue
            figure, axes, slider, save_button = make_window(starting_threshold)
            figure.suptitle(f"{name}   (close the window for the next photo)")

            def redraw(_=None, photo_parts=(straight_photo, top_down, scores)):
                draw_panels(axes, *photo_parts, slider.val, board_calibration)
                figure.canvas.draw_idle()

            slider.on_changed(redraw)
            redraw()
            plt.show()
            starting_threshold = slider.val  # keep the slider where it was for the next photo
        return 0

    with open("config/hardware.yaml") as f:
        camera_settings = yaml.safe_load(f)["camera"]
    live_camera = camera.open_camera(camera_settings)
    figure, axes, slider, save_button = make_window(starting_threshold)
    figure.suptitle("live (s = save raw photo, close window to quit)")
    latest = {}

    def save_on_s(event):
        if event.key == "s" and args.save and "photo" in latest:
            os.makedirs(args.save, exist_ok=True)
            path = os.path.join(args.save, time.strftime("%Y%m%d_%H%M%S") + ".jpg")
            cv2.imwrite(path, latest["photo"])
            print(f"saved {path}")

    figure.canvas.mpl_connect("key_press_event", save_on_s)
    plt.ion()
    try:
        while plt.fignum_exists(figure.number):
            latest["photo"] = camera.take_photo(live_camera, camera_settings["flush_frames"])
            straight_photo, top_down, scores = look_at_board(latest["photo"], *calibration)
            draw_panels(axes, straight_photo, top_down, scores, slider.val, board_calibration)
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        live_camera.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
