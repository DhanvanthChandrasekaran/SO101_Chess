"""Pick the camera's exposure and white balance by eye, then save them into config/hardware.yaml.

Usage: python calibration/camera_exposure.py   (needs the camera; close Cheese/guvcview first)
Set up the final lighting first. Press Auto for a starting point, adjust the sliders until the
"blown out" share is about 0% and dark pieces on dark squares are still visible, then press Save.
Flicker under mains lights: use exposure steps of 100 (50 Hz mains) or about 83 (60 Hz mains).
"""
import argparse
import os
import re
import subprocess
import sys
import time

import cv2
import matplotlib.pyplot as plt
import yaml
from matplotlib.widgets import Button, Slider

# Run as a script from the repo root, so make the project folders importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision import camera  # noqa: E402

HARDWARE_CONFIG = "config/hardware.yaml"
EXPOSURE = "exposure_time_absolute"
WHITE_BALANCE = "white_balance_temperature"
AUTO_SETTLE_S = 3.0           # how long the camera gets to choose its own values after Auto
PREVIEW_WIDTH_PX = 864        # the live picture is shrunk to this width so the window refreshes quickly
BLOWN_OUT_LEVEL = 250         # a pixel this bright (0-255) in any colour has lost detail
CRUSHED_LEVEL = 5             # a pixel this dark in every colour has lost detail


def run_v4l2(device: str, *arguments: str) -> str:
    """Run v4l2-ctl on the camera and return what it printed. Raises RuntimeError if it fails."""
    result = subprocess.run(["v4l2-ctl", "-d", device, *arguments], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"v4l2-ctl {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout


def read_camera_controls(device: str) -> dict:
    """Return every camera setting as {name: {"min": .., "max": .., "step": .., "value": ..}} from v4l2-ctl."""
    controls = {}
    for line in run_v4l2(device, "--list-ctrls").splitlines():
        match = re.match(r"\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:\s*(.*)", line)
        if match:
            name, _, details = match.groups()
            controls[name] = {key: int(number) for key, number in re.findall(r"(\w+)=(-?\d+)", details)}
    return controls


def lock(device: str, exposure: int, white_balance: int) -> None:
    """Switch off the automatic modes, then fix both values (two calls: some cameras ignore values while on auto)."""
    run_v4l2(device, "-c", "auto_exposure=1,white_balance_automatic=0")
    run_v4l2(device, "-c", f"{EXPOSURE}={exposure},{WHITE_BALANCE}={white_balance}")


def save_to_config(exposure: int, white_balance: int) -> None:
    """Replace only the exposure and white_balance values in config/hardware.yaml, keeping everything else."""
    with open(HARDWARE_CONFIG) as f:
        text = f.read()
    note = "# set by calibration/camera_exposure.py"
    for key, value in (("exposure", exposure), ("white_balance", white_balance)):
        text, count = re.subn(rf"^(\s+{key}:).*$", rf"\g<1> {value}  {note}", text, flags=re.MULTILINE)
        if count != 1:
            raise RuntimeError(f"expected one '{key}:' line in {HARDWARE_CONFIG}, found {count}")
    with open(HARDWARE_CONFIG, "w") as f:
        f.write(text)
    print(f"Saved to {HARDWARE_CONFIG}: exposure: {exposure}, white_balance: {white_balance}")


def main(command_line_args: list[str] | None = None) -> int:
    """Run the live adjustment window described at the top of this file."""
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(
        command_line_args)
    with open(HARDWARE_CONFIG) as f:
        camera_settings = yaml.safe_load(f)["camera"]
    device = camera_settings["device"]

    # open_camera refuses to run without saved values, so the camera is opened directly here.
    live_camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not live_camera.isOpened():
        print(f"Could not open camera {device} (is it plugged in, and is another app using it?)")
        return 1
    live_camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*camera_settings["fourcc"]))
    live_camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_settings["width"])
    live_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_settings["height"])

    state = {"moving_sliders_myself": False, "saved": False}
    try:
        controls = read_camera_controls(device)
        for needed in (EXPOSURE, WHITE_BALANCE, "auto_exposure", "white_balance_automatic"):
            if needed not in controls:
                print(f"The camera has no '{needed}' setting; send me the output of: v4l2-ctl -d {device} --list-ctrls")
                return 1
        other_automatic = [name for name in controls
                           if ("auto" in name or "dynamic" in name or name == "gain") and name not in ("auto_exposure", "white_balance_automatic")]
        if other_automatic:
            print("Other automatic settings that may also change the picture (tell me about these):")
            for name in other_automatic:
                print(f"  {name} = {controls[name].get('value')}")

        exposure_range, white_balance_range = controls[EXPOSURE], controls[WHITE_BALANCE]
        start_exposure = camera_settings.get("exposure") or exposure_range["value"]
        start_white_balance = camera_settings.get("white_balance") or white_balance_range["value"]
        lock(device, start_exposure, start_white_balance)

        figure = plt.figure(figsize=(11, 9))
        picture_axes = figure.add_axes([0.05, 0.25, 0.9, 0.7])
        picture_axes.axis("off")
        exposure_slider = Slider(figure.add_axes([0.2, 0.15, 0.6, 0.03]), "exposure", exposure_range["min"],
                                 exposure_range["max"], valinit=start_exposure, valstep=exposure_range.get("step", 1))
        white_balance_slider = Slider(figure.add_axes([0.2, 0.10, 0.6, 0.03]), "white balance",
                                      white_balance_range["min"], white_balance_range["max"],
                                      valinit=start_white_balance, valstep=white_balance_range.get("step", 1))
        auto_button = Button(figure.add_axes([0.3, 0.02, 0.15, 0.05]), "Auto")
        save_button = Button(figure.add_axes([0.55, 0.02, 0.15, 0.05]), "Save")

        def slider_moved(_):
            if not state["moving_sliders_myself"]:
                lock(device, int(exposure_slider.val), int(white_balance_slider.val))

        def auto_pressed(_):
            print(f"Letting the camera choose for {AUTO_SETTLE_S:.0f} s ...")
            run_v4l2(device, "-c", "auto_exposure=3,white_balance_automatic=1")
            settle_until = time.time() + AUTO_SETTLE_S
            while time.time() < settle_until:  # the camera only adjusts while frames are being taken
                live_camera.grab()
            chosen = read_camera_controls(device)
            exposure, white_balance = chosen[EXPOSURE]["value"], chosen[WHITE_BALANCE]["value"]
            print(f"Camera chose exposure {exposure}, white balance {white_balance}; locked there.")
            state["moving_sliders_myself"] = True
            exposure_slider.set_val(exposure)
            white_balance_slider.set_val(white_balance)
            state["moving_sliders_myself"] = False
            lock(device, exposure, white_balance)

        def save_pressed(_):
            save_to_config(int(exposure_slider.val), int(white_balance_slider.val))
            state["saved"] = True

        exposure_slider.on_changed(slider_moved)
        white_balance_slider.on_changed(slider_moved)
        auto_button.on_clicked(auto_pressed)
        save_button.on_clicked(save_pressed)

        picture_on_screen = None
        plt.ion()
        while plt.fignum_exists(figure.number):
            photo = camera.take_photo(live_camera, camera_settings["flush_frames"])
            brightest_colour = photo.max(axis=2)
            blown_out_percent = 100 * (brightest_colour >= BLOWN_OUT_LEVEL).mean()
            crushed_percent = 100 * (brightest_colour <= CRUSHED_LEVEL).mean()
            preview_height = round(photo.shape[0] * PREVIEW_WIDTH_PX / photo.shape[1])
            preview = cv2.cvtColor(cv2.resize(photo, (PREVIEW_WIDTH_PX, preview_height)), cv2.COLOR_BGR2RGB)
            if picture_on_screen is None:
                picture_on_screen = picture_axes.imshow(preview)
            else:
                picture_on_screen.set_data(preview)
            picture_axes.set_title(f"blown out (pure white): {blown_out_percent:.2f}%   "
                                   f"crushed (pure black): {crushed_percent:.2f}%   ({photo.shape[1]}x{photo.shape[0]})")
            plt.pause(0.05)
    except (RuntimeError, KeyboardInterrupt) as problem:
        print(problem)
    finally:
        live_camera.release()
    if not state["saved"]:
        print("Closed without saving; config/hardware.yaml is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
