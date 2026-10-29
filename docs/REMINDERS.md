# Reminders per subsystem
Open questions and checks that were deferred until the subsystem is built.
**When starting a subsystem, read its section and raise every unchecked item with the user before coding.** Tick items off (`[x]`) once they're resolved.

## Milestones (work top to bottom; each ends with a check that must pass before moving on)
| # | Build | Done when |
|---|---|---|
| M1 | `chess_logic/move_resolver.py` + `engine.py` + tests | `pytest tests/test_chess_logic.py` is green (quiet, capture, both castles, en passant, promotion, ambiguous, noise); engine smoke test returns a legal move |
| M2 | `tools/collect_states.py` (dataset capture) | `data/images/states/` has an empty-board shot, about 20 random states and at least 1 continuous game, each with a JSON label |
| M3 | `vision/camera.py` + `calibration/camera_intrinsics.py` | reprojection error < 0.5 px; board edges look straight in the undistorted image |
| M4 | `vision/board_view.py` + `calibration/board_corners.py` | the 8x8 grid overlay sits on the square edges in the top-down image, with a1 at the bottom-left; `empty_board.png` saved |
| M5 | `vision/occupancy.py` + `tools/vision_live.py` | 100% of squares correct on every M2 state, and `match_move` recovers every move of the M2 continuous game |
| M6 | `control/kinematics.py` + tests (URDF) | IK→FK round-trip passes for all 64 squares and the graveyard; unreachable squares are listed (move the board if any) |
| M7 | `control/arm.py` MockArm + `control/motion.py` | `tests/hardware/pick_place.py e2 e4 --mock` prints a sane, speed-limited trajectory; out-of-workspace targets raise |
| M8 | real `Arm` (no board under it) | read() matches teleop; small send() moves are slow; Ctrl-C parks and then disables torque |
| M9 | `calibration/arm_board.py` | residuals < 3 mm; a hover-only sweep puts the TCP over every square center |
| M10 | real pick/place | 20/20 quiet moves, plus a capture to the graveyard and castling |
| M11 | `main.py` with `--mock --images` | replays the M2 continuous game end to end, including one forced mismatch → retry → ask human |
| M12 | full real game | one full game, with verify/retry working |
M3-M5 (vision) and M6-M7 (kinematics) don't depend on each other.

## Environment (anytime)
- [ ] **ROS breaks pytest.** `~/.bashrc` line 134 runs `source /opt/ros/humble/setup.bash`, which exports `PYTHONPATH=/opt/ros/humble/...` (Python 3.10 packages). The `lerobot` env is Python 3.12, but Python still searches those ROS folders. pytest then auto-loads the ROS `launch_testing` plugin, which crashes (`No module named 'lark'`). Lerobot imports can pick up ROS packages too.
  Fix (user's choice): run `conda env config vars set PYTHONPATH= -n lerobot`, then `conda deactivate && conda activate lerobot`. This clears PYTHONPATH only while `lerobot` is active, and ROS still works in other shells. Until then, use `PYTHONPATH= pytest`.
- [ ] **placo install decision** (for lerobot IK). Command: `pip install "placo>=0.9.6,<0.9.16" "cmeel-urdfdom>=4,<5" "cmeel-tinyxml2<11"` (the pins lerobot uses for its `kinematics` extra). A dry run on 2026-10-29 showed 35 new packages (placo 0.9.15, pin/Pinocchio 3.4.0, cmeel-boost, eigenpy, meshcat, ipython, ...), and none of numpy, opencv or torch is replaced.
- [ ] **Is `data/test_images/` committed?** It's about 430 KB and not gitignored. `data/images/` and `data/logs/` are gitignored.

## chess_logic/move_resolver.py and main.py (build step 2 and 7)
- [ ] **Ambiguous human captures.** Vision only sees "occupied or empty". When the human captures, the target square was occupied before and is still occupied after, so the only visible change is the **from-square** emptying. If the moving piece could legally capture on more than one square, every option looks identical. For example, a queen on d1 that can take on d7 or on g4: both just "empty d1". So `match_move` returns all candidates, and `main` asks the human to type the move (e.g. `d1g4`). Decide: is typing OK, or should `main` show a numbered list to pick from?
- [ ] Human promotion is assumed to be a queen (vision can't tell). Robot promotion: the human swaps the pawn for a queen when prompted.

## vision/camera.py (build step 3)
- [x] Device found: `/dev/v4l/by-id/usb-Xitech_USB_Camera_20250606105-video-index0`. MJPG modes up to 2592x1944 at 30 fps; YUYV is slow at high resolution.
- [ ] Confirm 2592x1944 vs 1920x1080: does the 16:9 mode crop the field of view? Compare two captures. Close Cheese/browser tabs first, because only one app can stream at a time.
- [ ] How exposure and white balance are locked: v4l2-ctl or OpenCV properties, and the values.

## calibration/camera_intrinsics.py
- [ ] Checkerboard: inner-corner count, square size in mm. Fisheye or pinhole model (fisheye is probably better at 106°).

## calibration/board_corners.py and vision/board_view.py
- [ ] Click order is a1, h1, h8, a8 (outer corners of the playing area), so image orientation doesn't matter. The user assumed a1 is at the bottom-left of the camera image; this is only for sanity-checking the clicks.
- [ ] The **empty-board reference photo** (user is getting these): same camera pose, light and exposure as play, and the arm parked out of view.
- [ ] The test images may be mirrored/flipped (user note). In `data/test_images/` the a-h letters read normally along the bottom with white at the bottom, but double-check before using them as ground truth.

## vision/occupancy.py
- [ ] Threshold tuning on saved images. Watch for: the folding-board hinge seam between the d and e files, glare on light squares, dark pieces on dark squares (all visible in the test images).
- [ ] The arm (orange) is in view at the bottom of Test Image 2. The park pose must keep the arm and its shadow off the board during captures.

## control/kinematics.py (build step 4)
- [ ] Download `so101_new_calib.urdf` (SO-ARM100 repo, `Simulation/SO101/`), set `arm.yaml: urdf_path`, and document its base-frame axes in INTERFACES.md §2.7.
- [ ] Check that lerobot 0° matches URDF zero on the real arm (FK of the teleop pose vs a tape measure). If it's off, either add per-joint offsets or recalibrate with `lerobot-calibrate`.
- [ ] `tcp_offset_m`: from `gripper_frame_link` to the end/middle of the open jaws.
- [ ] **Reach**: the far corners (a8/h8) may be at the limit. The user will move the board closer if needed. Run the IK test on all 64 squares first.
- [ ] Pitch schedule values (`pitch_near_deg`, `pitch_far_deg`, `reach_near_m`, `reach_far_m`) and `wrist_roll_deg`.

## control/arm.py (build step 5)
- [ ] Gripper 0-100: which end is open; `gripper_open`/`gripper_close` values.
- [ ] Gripper force: closing past the piece with `gripper_torque` (Max_Torque_Limit). lerobot also sets `Overload_Torque=25%` on the gripper, so check the grip doesn't weaken during a long stall (a carried king slipping).
- [ ] Joint limits per joint (from teleop).
- [ ] The user plans to recalibrate the follower (currently `HAM_follower_001`).

## control/motion.py
- [ ] **`travel_height_m` must clear a king while carrying a king**: carried piece bottom = travel z − grip z. With grip at 0.7 × 4.9 cm ≈ 3.4 cm, travel needs > 4.9 + 3.4 + margin ≈ 9-10 cm. `hover_height_m: 0.07` is "above the piece top", so it's fine.
- [ ] `max_joint_speed_deg_s`, plus the graveyard spot (`board.yaml: graveyard_xy_m`, board frame, in reach).
- [ ] Piece heights for knight, bishop, rook and queen (`board.yaml`).

## calibration/arm_board.py (build step 6)
- [ ] Teleop squares for the sparse offsets: start with a1, h1, h8, a8, plus d4/e5. Add more later with `--squares`.
- [ ] Record the park pose: out of the camera's view of the board, and safe to drop to when torque is disabled.
