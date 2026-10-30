# Reminders per subsystem
Open questions and checks that were deferred until the subsystem is built.
**When starting a subsystem, read its section and raise every unchecked item with the user before coding.** Tick items off (`[x]`) once they're resolved.

## Milestones (agreed 2026-10-29; work in order, each ends with a check that must pass)
Order: 1, then 2 and 3 in either order (independent), then 4, then 5.

| # | Milestone | Files | Done when |
|---|---|---|---|
| 1 | **Chess brain**: work out the human's move from before/after occupancy; Stockfish reply; reply turned into move/remove steps for the arm | `chess_logic/*`, `tests/test_chess_logic.py` | `pytest` green: quiet, capture, both castles, en passant, promotion, ambiguous capture, misread/noise; engine smoke test |
| 2 | **Robot can see**: labelled-photo collection script (random positions + move-by-move game + empty board), intrinsics/undistort, corner clicks + top-down view, occupancy, live overlay | `tools/collect_states.py` (approved new file, Sonnet writes it here), `vision/*`, `calibration/camera_intrinsics.py`, `calibration/board_corners.py`, `tools/vision_live.py` | 100% of squares correct on all collected photos; the recorded game's moves are all recovered by `find_human_move` (ambiguous captures may return several candidates); live overlay looks right |
| 3 | **Arm in simulation**: lerobot IK on the URDF, pick/place sequences, MockArm | `control/kinematics.py`, `control/motion.py`, `control/arm.py` (MockArm), `tests/test_kinematics.py` | valid IK for all 64 squares + graveyard (or a list of unreachable ones, so the board can be moved); `tests/hardware/pick_place.py e2 e4 --mock` prints a smooth, slow trajectory |
| 4 | **Arm on hardware**: real driver, safety (limits, slow speed, Ctrl-C parks then goes limp), teleop arm-to-board calibration, real pick/place | `control/arm.py` (Arm), `calibration/arm_board.py`, `tests/hardware/pick_place.py` | 20/20 moves without drops, plus a capture to the graveyard (special moves are done by the human, INTERFACES §2.13) |
| 5 | **Full game**: `main.py` loop with verify/retry/ask-human and logging; first `--mock --images` replay of the recorded game, then real | `main.py` | replay passes (including a forced mismatch → retry → ask human, a recorded ambiguous capture → ambiguity prompt → typed answer, and a robot special move made by the human); one full real game |

## Environment (anytime)
- [x] **ROS breaks pytest.** (Fixed 2026-10-29: `PYTHONPATH` is cleared in the `lerobot` env.) `~/.bashrc` line 134 runs `source /opt/ros/humble/setup.bash`, which exports `PYTHONPATH=/opt/ros/humble/...` (Python 3.10 packages). The `lerobot` env is Python 3.12, but Python still searches those ROS folders. pytest then auto-loads the ROS `launch_testing` plugin, which crashes (`No module named 'lark'`). Lerobot imports can pick up ROS packages too.
  Fix (user's choice): run `conda env config vars set PYTHONPATH= -n lerobot`, then `conda deactivate && conda activate lerobot`. This clears PYTHONPATH only while `lerobot` is active, and ROS still works in other shells. Until then, use `PYTHONPATH= pytest`.
- [x] **placo install decision** (Installed by the user 2026-10-29: placo 0.9.15, pin 3.4.0; numpy/opencv unchanged.) (for lerobot IK). Command: `pip install "placo>=0.9.6,<0.9.16" "cmeel-urdfdom>=4,<5" "cmeel-tinyxml2<11"` (the pins lerobot uses for its `kinematics` extra). A dry run on 2026-10-29 showed 35 new packages (placo 0.9.15, pin/Pinocchio 3.4.0, cmeel-boost, eigenpy, meshcat, ipython, ...), and none of numpy, opencv or torch is replaced.
- [ ] **Is `data/test_images/` committed?** It's about 430 KB and not gitignored. `data/images/` and `data/logs/` are gitignored.

## chess_logic/move_resolver.py and main.py (milestones 1 and 5)
- [x] **Ambiguous human captures (decided).** When the human captures, occupancy only shows the from-square emptying, so a queen that could take on two squares looks the same either way. `find_human_move` returns every candidate and `main` shows the friendly ambiguity prompt (INTERFACES.md, main.py section). Milestone 1 test: an ambiguous slider capture returns all candidates. Auto-resolving by change scores is removed (OPTIONAL_IMPROVEMENTS.md).
- [x] Human promotion is assumed to be a queen (vision can't tell). The robot's special moves (castling, en passant, promotion) are made by the human when prompted (INTERFACES §2.13).

## vision/camera.py (milestone 2)
- [x] Device found: `/dev/v4l/by-id/usb-Xitech_USB_Camera_20250606105-video-index0`. MJPG modes up to 2592x1944 at 30 fps; YUYV is slow at high resolution.
- [x] 2592x1944 is the full field of view (about 20 fps). 1920x1080 is a crop, so don't use it. Only one app can stream at a time, so close Cheese first.
- [x] **The printed board labels are rotated 180° from our convention.** In the mounted setup (white at the robot's side), the printed "h8" corner is our a1. Ignore the printed letters: a1 = the dark square nearest the robot, on its left (bottom-left of the camera image). Confirmed by the user; `data/images/probe/probe_2592x1944.jpg` shows the dark a1 at the bottom-left with the robot at the bottom.
- [ ] Exposure and white balance: locked with `v4l2-ctl` (decided). Pick the values with `python calibration/camera_exposure.py` under the final lighting (Auto, adjust, Save); it writes them into `hardware.yaml`. Check the other automatic settings it lists (gain, autofocus) and raise them.

## calibration/camera_intrinsics.py
- [ ] Checkerboard shown on a screen (no printer): `python calibration/camera_intrinsics.py --make-checkerboard checkerboard.png --screen-px W H`, opened full screen at 100% zoom. Measure across all squares with a ruler, divide by the number of squares, and set `vision.yaml: checkerboard.square_mm` (TODO). Prefer the laptop (bigger board in the photo). Turn brightness up and tilt to avoid glare; don't let the screen sleep. Lens model: fisheye (decided).
- [x] Taking the checkerboard photos: `python calibration/camera_intrinsics.py --capture data/images/checkerboard/` (automatic). Pressing `s` in `tools/vision_live.py --no-lens --save DIR` also works.

## calibration/board_corners.py and vision/board_view.py
- [ ] Click order is a1, h1, h8, a8 (outer corners of the playing area), so image orientation doesn't matter. The user assumed a1 is at the bottom-left of the camera image; this is only for sanity-checking the clicks.
- [ ] The **empty-board reference photo** (user is getting these): same camera pose, light and exposure as play, and the arm parked out of view.
- [x] The test images are mirrored left to right (printed letters read backwards; checked 2026-10-29). Flip them before using them as ground truth. In `data/test_images/` the a-h letters read normally along the bottom with white at the bottom, but double-check before using them as ground truth.

## vision/occupancy.py
- [ ] Threshold tuning on saved images. Watch for: the folding-board hinge seam between the d and e files, glare on light squares, dark pieces on dark squares (all visible in the test images).
- [ ] The arm (orange) is in view at the bottom of Test Image 2. The park pose must keep the arm and its shadow off the board during captures.

## control/kinematics.py (milestone 3)
- [ ] Download `so101_new_calib.urdf` (SO-ARM100 repo, `Simulation/SO101/`), set `arm.yaml: urdf_path`, and document its base-frame axes in INTERFACES.md §2.7.
- [ ] Check that lerobot 0° matches URDF zero on the real arm (FK of the teleop pose vs a tape measure). If it's off, either add per-joint offsets or recalibrate with `lerobot-calibrate`.
- [ ] `tcp_offset_m`: from `gripper_frame_link` to the end/middle of the open jaws.
- [ ] **Reach**: the far corners (a8/h8) may be at the limit. The user will move the board closer if needed. Run the IK test on all 64 squares first.
- [ ] Pitch schedule values (`pitch_near_deg`, `pitch_far_deg`, `reach_near_m`, `reach_far_m`) and `wrist_roll_deg`.

## control/arm.py (milestones 3-4)
- [ ] Gripper 0-100: which end is open; `gripper_open`/`gripper_close` values.
- [ ] Gripper force: closing past the piece with `gripper_torque` (Max_Torque_Limit). lerobot also sets `Overload_Torque=25%` on the gripper, so check the grip doesn't weaken during a long stall (a carried king slipping).
- [ ] Joint limits per joint (from teleop).
- [ ] The user plans to recalibrate the follower (currently `HAM_follower_001`).

## control/motion.py
- [ ] **`travel_height_m` must clear a king while carrying a king**: carried piece bottom = travel z − grip z. With grip at 0.7 × 4.9 cm ≈ 3.4 cm, travel needs > 4.9 + 3.4 + margin ≈ 9-10 cm. `hover_height_m: 0.07` is "above the piece top", so it's fine.
- [ ] `max_joint_speed_deg_s`, plus the graveyard spot (`board.yaml: graveyard_xy_m`, board frame, in reach).
- [ ] Piece heights for knight, bishop, rook and queen (`board.yaml`).

## calibration/arm_board.py (milestone 4)
- [ ] Teleop squares for the sparse offsets: start with a1, h1, h8, a8, plus d4/e5. Add more later with `--squares`.
- [ ] Record the park pose: out of the camera's view of the board, and safe to drop to when torque is disabled.
