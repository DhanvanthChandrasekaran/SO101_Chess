# Interfaces
Source of truth for what data moves between modules. Written/updated BEFORE code changes.
Status: v3 (plain-language names, CLAUDE.md rule 13).

## 1. Data flow (one line per hop)
Only `main.py` calls across modules. `motion.py` is the one module that calls others (`kinematics`, `arm`), as CLAUDE.md allows.

```
startup
  main -> camera.load_lens_calibration(path)                          -> lens_calibration dict
  main -> board_view.load_board_calibration(path)                     -> board_calibration dict
  main -> kinematics.load_arm_calibration(folder, arm_settings, mock) -> arm_calibration dict
  main -> engine.open_engine(engine_cfg)                              -> stockfish (chess.engine.SimpleEngine)
  main -> camera.open_camera(camera_settings)                         -> camera (cv2.VideoCapture)
  main -> Arm(hardware_settings, arm_settings) | MockArm(...)  .connect()
  main -> motion.park(arm, settings)

look at the board (after every move, human or robot)
  main -> camera.take_photo(camera, old_frames_to_skip)              -> photo          (H,W,3) uint8 BGR
  main -> camera.undistort(photo, lens_calibration)                  -> straight photo (H,W,3) uint8 BGR
  main -> board_view.top_down_view(straight_photo, board_calibration) -> top_down      (N,N,3) uint8 BGR
  main -> occupancy.square_difference_scores(top_down, empty_board_top_down, vision_settings) -> scores (8,8) float32
  main -> occupancy.occupied_squares(scores, vision_settings)        -> occupied       (8,8) bool

human turn (after spacebar)
  main -> move_resolver.find_human_move(board, before, after)        -> list[chess.Move]  (0, 1 or many; many -> ambiguity prompt)

robot turn
  main -> engine.best_move(stockfish, board, engine_cfg)             -> chess.Move
  if move is special (§2.13): main prints it, the human makes it on the board, presses spacebar -> go to check
  else:
  main -> move_resolver.get_robot_move(board, move)                  -> steps     list of ("move", start, end) / ("remove", square)
  main -> motion.plan_waypoints(steps, piece_heights, settings)      -> waypoints list of ("go_to", joint angles) / ("gripper", "open"|"close")
          motion -> kinematics.square_center_on_board / board_point_to_robot / gripper_tilt_for / joint_angles_for
  main -> motion.run_waypoints(arm, waypoints, settings)             -> None
          motion -> arm.read_joint_angles() / arm.set_joint_targets() / arm.set_gripper_strength()
  main -> motion.park(arm, settings)                                 -> None   (arm out of camera view)

check (after robot move, and at startup)
  main -> move_resolver.expected_board_state(board)                  -> (8,8) bool; main compares with `occupied`

stop (Ctrl-C anywhere)
  main -> motion.park(arm, settings) -> arm.turn_off_motors() -> arm.disconnect()   (second Ctrl-C: turn_off_motors at once)
```

## 2. Conventions (defined once, used everywhere)

### 2.1 Game orientation
- **Robot plays white and moves first** (`config/game.yaml: robot_color: white`). For now, any other value raises `NotImplementedError` at startup.
- Robot sits at the rank-1 edge, centered. Human sits opposite, at the rank-8 edge.
- **Physical frame = "robot-white" frame.** Every physical calibration (corner clicks, teleop squares) uses square names as if the robot plays white. To support black later, `main` rotates the 8x8 grid 180° (`sq ^ 63`) between chess squares and physical squares. Nothing else changes. See `docs/OPTIONAL_IMPROVEMENTS.md`.

### 2.2 Squares
- A square is a python-chess number 0-63: `square = rank_index*8 + file_index`, with `file_index` 0=a..7=h and `rank_index` 0=rank 1..7=rank 8. So a1=0, h1=7, e2=12, e4=28, a8=56, h8=63.
- Square names appear only in user-facing text and the CLI (`chess.square_name`, `chess.parse_square`).

### 2.3 Occupancy grid
- `np.ndarray`, shape `(8, 8)`, dtype `bool`, indexed `occupied[rank_index, file_index]`. `occupied[0, 0]` is a1 and `occupied[7, 7]` is h8.
- Therefore `occupied.reshape(64)[square]` is the occupancy of python-chess square `square`.
- `True` = occupied. Difference scores (`float32`, same layout) are higher when the square is more likely occupied.
- This is **not** image layout. To display it with rank 8 on top, use `np.flipud(occupied)`.

### 2.4 Image pixel frame (raw and undistorted)
- `np.ndarray (H, W, 3) uint8` in BGR (OpenCV) order, indexed `[v, u]`. `u` runs right and `v` runs down, origin at the top-left pixel, units px.
- Raw and undistorted frames have the same size `(H, W)` = camera resolution from `config/hardware.yaml`.

### 2.5 Top-down board image
- `np.ndarray (N, N, 3) uint8` BGR, where `N = 8 * pixels_per_square`. `pixels_per_square` is stored in `calibration/data/board.yaml`, copied there from `config/vision.yaml: px_per_square` when the calibration was made.
- It covers exactly the 8x8 playing area, with no border. **a1 is at the bottom-left and h8 at the top-right**, so the image looks like the board seen from the robot's (white's) side.
- The center of square (file_idx f, rank_idx r) is at `u = (f + 0.5)*p` and `v = (7 - r + 0.5)*p`, with `p = pixels_per_square`.
- It is a homography of the **board plane** (z = 0). Tall pieces lean outward, so the occupancy check samples only a centered patch at each square center.

### 2.6 Board plane frame (meters)
- Origin at the **outer corner of a1** of the 26.5 cm playing area. +x runs along rank 1 toward the h-file, +y along the a-file toward rank 8, +z points up, and z = 0 is the board surface.
- Square center: `x = (f + 0.5)*s`, `y = (r + 0.5)*s`, with `s = square_size_m` = 0.265/8 = 0.033125 m. Example: e2 is at (0.1490625, 0.0496875).
- Off-board spots (graveyard, spare pieces) are also given in this frame, e.g. `x < 0` or `y < 0`.

### 2.7 Robot base frame (meters)
- This is the **world frame of lerobot `RobotKinematics`**, which is the `base` link of the SO101 URDF `so101_new_calib.urdf` (SO-ARM100 repo). Kinematics uses lerobot's IK/FK (placo). A custom IK is in `docs/OPTIONAL_IMPROVEMENTS.md`.
- Axis directions and origin: **TODO, documented here once the URDF is downloaded**. Expected: origin at the base mount, +z up. Which horizontal axis points at the board is read from the URDF.
- With the robot playing white, **a1 is front-left of the robot and h1 is front-right**.
- `board_to_robot_transform` (4x4, float64, homogeneous) maps board-frame points to base-frame points. It is fitted by `calibration/arm_board.py`, so a rough setup is fine and the axis convention doesn't affect correctness.

### 2.8 Joint angles
- Joint dicts at every boundary are `dict[str, float]` with bare lerobot names: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`. `arm.py` adds and strips the `.pos` suffix.
- The 5 body joints are in **lerobot SO101 degrees** (`use_degrees=True`). 0° is the middle of the range recorded by `lerobot-calibrate` (calibration id `HAM_follower_001`).
- `so101_new_calib.urdf` is built for this convention (URDF joint zero = lerobot 0°). So **lerobot degrees go straight into `RobotKinematics`** with no zero offsets. Any leftover mismatch shows up as residuals in `arm_board.py` and is absorbed by the per-square offsets. Check it on the bench (see REMINDERS).
- `gripper` is lerobot `RANGE_0_100`. Which end is open is TODO, to be checked on the bench. It is never passed to IK.

### 2.9 Tool pitch
- Degrees from horizontal: positive points downward, 90 = straight down.
- It is not solved for. It follows a config schedule linear in horizontal reach `ρ = hypot(x, y)` in the base frame: `pitch_near_deg` at `ρ_near_m` and `pitch_far_deg` at `ρ_far_m`, clamped outside that range. With both values equal, this becomes a constant.
- `wrist_roll` is a fixed config constant (`wrist_roll_deg`).
- Inside `kinematics.joint_angles_for`, the desired 4x4 TCP pose is built from the position, yaw `atan2(y, x)` (the 5-DOF arm can't choose its yaw), the pitch, and the fixed roll. lerobot's IK then solves it with orientation weight `ik_orientation_weight`. The result is accepted only if FK reproduces it within `ik_pos_tol_m` and `ik_pitch_tol_deg`.

### 2.10 Other units
- Lengths are in meters. Times are in seconds. Angles at boundaries are in degrees.
- The gripper torque limit is in raw Feetech `Max_Torque_Limit` units, 0-1000. lerobot's default for the gripper is 500.

### 2.11 Steps and waypoints
- Step (chess_logic -> control, from `get_robot_move`): `("move", start_square, end_square)` or `("remove", square)`. Squares are python-chess ints.
- Waypoint (made by `motion.plan_waypoints`, public for debugging): `("go_to", dict[str, float])` (5 body joints, degrees) or `("gripper", "open" | "close")`.

### 2.12 Settings dicts
`main` loads each YAML once and passes plain dicts in. Modules never read `config/` themselves. A parameter named `<x>_settings` is the dict from `config/<x>.yaml` (`camera_settings` is the `camera:` part of `hardware.yaml`; `engine_cfg` is the `engine:` part of `game.yaml`).
Files: `hardware.yaml` (camera, arm port and id), `arm.yaml` (kinematics, tool, motion, gripper, safety), `board.yaml` (board, pieces, off-board spots), `vision.yaml` (lens calibration settings, top-down view and occupancy tuning), `game.yaml` (colour, first move, start FEN, Stockfish).
`settings` (motion) = `{"arm": arm.yaml dict, "board": board.yaml dict, "arm_calibration": dict from load_arm_calibration}`.

### 2.13 Who moves the robot's pieces
- The **arm** executes the robot's normal moves: quiet moves and ordinary captures (captured piece to the graveyard, then the move).
- **Special moves are made physically by the human** when prompted: castling, en passant and promotion. `main` decides with python-chess: `board.is_castling(m) or board.is_en_passant(m) or m.promotion is not None`. It prints the move (SAN and UCI; for promotion, "replace the pawn with a queen", adding "a captured queen, or any stand-in piece if none is off the board", since there is only one set), waits for spacebar, then runs the normal verify step against `expected_board_state`.
- `get_robot_move` still returns full steps for special moves (unchanged contract and tests), so handing them to the arm later only changes `main`. See `docs/OPTIONAL_IMPROVEMENTS.md`.

## 3. Function contracts

### vision/camera.py
| | |
|---|---|
| `open_camera(camera_settings: dict) -> cv2.VideoCapture` | Opens `camera_settings["device"]` (a `/dev/v4l/by-id/...` path) at `width`x`height`, then locks exposure and white balance to the config values with `v4l2-ctl` (manual exposure, auto white balance off). **Raises `RuntimeError`** if the camera can't be opened, the resolution is refused, `exposure` or `white_balance` is still null in config, or `v4l2-ctl` fails. |
| `take_photo(camera: cv2.VideoCapture, old_frames_to_skip: int) -> np.ndarray` | Reads and discards `old_frames_to_skip` buffered frames (`hardware.yaml: flush_frames`), then returns a fresh `(H,W,3) uint8 BGR` photo. **Raises `RuntimeError`** if the read fails. |
| `load_lens_calibration(path: str) -> dict` | Returns `{"camera_matrix": (3,3) float64, "distortion": (4,) or (5,) float64, "lens_model": "fisheye" or "pinhole", "image_size": (W, H) int}`. **Raises `FileNotFoundError`**. |
| `undistort(photo: np.ndarray, lens_calibration: dict) -> np.ndarray` | Takes a raw photo and returns an undistorted one with the same shape and dtype (`balance`/`alpha` = 0, so pixels outside the original are black). **Raises `ValueError`** if the photo size != `lens_calibration["image_size"]`. |

Example: `undistort(take_photo(camera, 3), lens_calibration)` gives `(1944, 2592, 3) uint8`.
Offline: use `cv2.imread("data/images/x.png")` instead of `take_photo`.

### vision/board_view.py
| | |
|---|---|
| `load_board_calibration(path: str) -> dict` | Returns `{"photo_to_top_down": (3,3) float64 homography (undistorted px -> top-down px), "pixels_per_square": int, "corner_pixels": (4,2) float64 [a1, h1, h8, a8] outer corners in undistorted px}`. **Raises `FileNotFoundError`**. |
| `top_down_view(photo: np.ndarray, board_calibration: dict) -> np.ndarray` | Takes an undistorted `(H,W,3) uint8` photo and returns the top-down `(N,N,3) uint8` image (§2.5). Never fails for a valid array. |

Example: with `pixels_per_square = 50`, `top_down_view(straight_photo, board_calibration).shape == (400, 400, 3)`, and the e2 center is at `[v=325, u=225]`.

### vision/occupancy.py
| | |
|---|---|
| `square_difference_scores(top_down: np.ndarray, empty_board_top_down: np.ndarray, vision_settings: dict) -> np.ndarray` | Compares a centered patch (`patch_frac` of the square side) of each square in `top_down` against the same patch of the empty-board reference (same shape). Returns `(8,8) float32`, ≥ 0, layout §2.3. **Raises `ValueError`** on shape mismatch. |
| `occupied_squares(difference_scores: np.ndarray, vision_settings: dict) -> np.ndarray` | Returns `difference_scores > vision_settings["threshold"]` as `(8,8) bool`. **Raises `ValueError`** while `threshold` is still null in config. |

Example: for the start position, `occupied_squares(...)` has rows 0, 1, 6 and 7 all True.
The method compares against an empty-board reference image, so it doesn't rely on piece colour. The scoring internals are an implementation detail.

### chess_logic/engine.py
| | |
|---|---|
| `open_engine(engine_cfg: dict) -> chess.engine.SimpleEngine` | Starts the binary at `engine_cfg["path"]` and sets `Skill Level` (0-20). **Raises `FileNotFoundError`** if the binary is missing, and `chess.engine.EngineError` if it fails to start. |
| `best_move(stockfish, board: chess.Board, engine_cfg: dict) -> chess.Move` | Thinks for `think_time_s` (time limit only). Does not modify `board`. **Raises `engine.GameOverError`** (subclass of `ValueError`) if `board.is_game_over()` (no draw claims: threefold/50-move do not count). If the engine crashes or fails, its `chess.engine.EngineError` / `EngineTerminatedError` **propagates** (no restart); `main` stops explicitly. |
| `close_engine(stockfish) -> None` | Quits the engine. Safe to call twice. |

Example: `best_move(stockfish, chess.Board(), cfg)` might return `Move.from_uci("e2e4")`.

### chess_logic/move_resolver.py
| | |
|---|---|
| `expected_board_state(board: chess.Board) -> np.ndarray` | `(8,8) bool`: True where python-chess has a piece. Never fails. |
| `find_human_move(board: chess.Board, before: np.ndarray, after: np.ndarray) -> list[chess.Move]` | Finds the human's move. `board` is the position before the move (unchanged). `before` and `after` are perceived `(8,8) bool`. Returns every legal move `m` for which the squares changed by `m` (from `expected_board_state`) equal `before ^ after`. Promotions to the same square are merged into the **queen** promotion. **Never raises.** `[]` means no match, 1 element means resolved, and more than one means ambiguous (a capture shows only its from-square emptying); `main` then shows the ambiguity prompt (see main.py below). |
| `get_robot_move(board: chess.Board, move: chess.Move) -> list[tuple]` | `board` is the position **before** `move` (unchanged). Returns steps (§2.11), with removals first. **Raises `ValueError`** if `move` is illegal. |

Comparing changed squares (`before ^ after`) instead of the whole grid means that a square vision consistently misreads doesn't break matching.

`find_human_move` example: with a white queen on d1 that can take on d7 or g4, occupancy shows only that d1 emptied, so it returns both `[Move.from_uci("d1d7"), Move.from_uci("d1g4")]`.

`get_robot_move` examples:
- Quiet move e2e4: `[("move", 12, 28)]`
- Capture e4xd5: `[("remove", 35), ("move", 28, 35)]`
- White castling O-O: `[("move", 4, 6), ("move", 7, 5)]` (king first, then rook)
- En passant e5xd6: `[("remove", 35), ("move", 36, 43)]`
- Promotion e7e8=Q: `[("move", 52, 60)]`.
- Castling, en passant and promotion are special moves (§2.13): today `main` has the human make them and does not send these steps to `motion`.

### control/kinematics.py (settings dicts + `lerobot.model.kinematics.RobotKinematics`)
| | |
|---|---|
| `load_arm_calibration(calibration_folder: str, arm_settings: dict, allow_rough_defaults: bool) -> dict` | Returns `{"arm_model": RobotKinematics (built from arm_settings["urdf_path"], target_frame), "board_to_robot_transform": (4,4) float64, "square_corrections": list[((x, y), (dx, dy))] (board m), "park_pose": dict[str, float] deg}`. If calibration files are missing: with `allow_rough_defaults` (only `--mock`) it uses `arm_settings["nominal"]`; otherwise it **raises `FileNotFoundError`**. A missing URDF always raises `FileNotFoundError`. If placo is missing, lerobot's `ImportError` propagates. |
| `square_center_on_board(square: int, board_settings: dict) -> tuple[float, float]` | Square center in the board frame, in meters (§2.6). **Raises `ValueError`** if `square` is not 0-63. |
| `board_point_to_robot(board_x_y: tuple[float, float], height_above_board: float, arm_calibration: dict) -> np.ndarray` | Board-frame `(x, y)` plus a height above the board gives `(3,) float64` in the robot base frame. It applies `board_to_robot_transform` plus the xy correction interpolated from `square_corrections` (inverse-distance weighting; zero if there are none). |
| `gripper_tilt_for(point_from_robot: np.ndarray, arm_settings: dict) -> float` | Degrees from horizontal (§2.9). |
| `joint_angles_for(point_from_robot: np.ndarray, gripper_tilt: float, starting_guess: dict[str, float], arm_settings: dict, arm_calibration: dict) -> dict[str, float] \| None` | TCP position (base frame, m) plus tilt gives the 5 body joints in degrees (§2.8). `starting_guess` is where the iterative solver starts: the current joints, or the previous waypoint. **Returns `None`** if the reverse check (`gripper_position_for`) fails (unreachable) or any joint is outside `joint_limits_deg`. |
| `gripper_position_for(joint_angles: dict[str, float], arm_settings: dict, arm_calibration: dict) -> tuple[np.ndarray, float]` | 5 body joints in degrees give the TCP `(3,) float64` in base m (target frame plus `tcp_offset_m`) and the tilt in degrees. Missing keys **raise `KeyError`**. |

Example: `point = board_point_to_robot(square_center_on_board(12, board_settings), 0.02, arm_calibration)`, then `gripper_position_for(joint_angles_for(point, gripper_tilt_for(point, arm_settings), arm_calibration["park_pose"], arm_settings, arm_calibration), arm_settings, arm_calibration)` gives back `≈ (point, tilt)` to within `ik_pos_tol_m` and `ik_pitch_tol_deg`.

### control/arm.py (custom classes crossing a boundary: approved)
`Arm(hardware_settings: dict, arm_settings: dict)` wraps lerobot `SO101Follower`. `MockArm(hardware_settings, arm_settings, starting_joint_angles: dict[str, float])` has the same methods, so the two can be swapped.
| | |
|---|---|
| `connect() -> None` | **Raises `ConnectionError`** if the port is missing (from lerobot). `MockArm` never fails. |
| `read_joint_angles() -> dict[str, float]` | All 6 joints (§2.8). |
| `set_joint_targets(targets: dict[str, float]) -> dict[str, float]` | Takes any subset of the 6 joints. Each value is **clamped** to `joint_limits_deg` / gripper 0-100, and the dict actually sent is returned. Never blocks for motion; `motion` interpolates. |
| `set_gripper_strength(torque_limit: int) -> None` | Writes the gripper `Max_Torque_Limit` (0-1000). When the gripper is commanded closed, it stalls on the piece at this force instead of needing a per-piece position. |
| `turn_off_motors() -> None` | Disables torque on all motors. Never raises (errors are logged). |
| `disconnect() -> None` | Safe to call twice. |

### control/motion.py
| | |
|---|---|
| `plan_waypoints(steps: list[tuple], piece_heights: dict[int, float], settings: dict) -> list[tuple]` | `steps` come from `get_robot_move`. `piece_heights[square]` is the height (m) of the piece standing on `square` **before** the move, for every square a step picks from. Returns waypoints (§2.11) covering the whole move: for each pick, hover, open, descend to `grip_height_frac*height`, close, lift to travel height; then above the destination, descend, open, lift. A `remove` goes to `graveyard_xy_m[0]`. Plans everything before anything moves. **Raises `ValueError`** if any target is outside the board workspace (+ `workspace_margin_m`, graveyard allowed), below `min_z_m`, or unreachable by `joint_angles_for`. |
| `run_waypoints(arm, waypoints: list[tuple], settings: dict) -> None` | Interpolates in joint space from `arm.read_joint_angles()` toward each waypoint, at `max_joint_speed_deg_s * max_speed_scale` and `control_hz`. Small ticks keep Ctrl-C responsive. **Raises `RuntimeError`** if the arm does not reach a waypoint within `settle_tol_deg` and `step_timeout_s`. |
| `park(arm, settings: dict) -> None` | Moves to `arm_calibration["park_pose"]` (via travel height) the same way `run_waypoints` does. |

Example: `plan_waypoints([("move", 12, 28)], {12: 0.028}, settings)` gives about 10 waypoints, starting with `("gripper", "open")` and a `("go_to", {...})` hover above e2.

### main.py (not imported by anyone)
`main(command_line_args: list[str] | None = None) -> int`, with flags `--mock` and `--images DIR` (offline frames). Exit code 0 means the game ended normally, 1 means stopped.
**Ambiguity prompt** (`find_human_move` returns more than one move). `main` builds it from python-chess (colour, piece name, squares):
```
I couldn't tell which capture you made.
Your black queen on d8 moved. It could have gone to:
  1. d8 → d2   capturing white pawn on d2
  2. d8 → h4   capturing white knight on h4
Type the move (e.g. d8h4) or just the destination square (e.g. h4):
```
Non-capture candidates have no "capturing" part. It accepts the full UCI move, the destination square, or the list number. Anything else prints "That isn't one of the options above" and asks again.

`piece_heights` is built in `main` from python-chess: `{square: board_settings["piece_height_m"][chess.piece_name(piece.piece_type)] for square, piece in board.piece_map().items()}`.

## 4. Calibration artifacts (`calibration/data/`, generated only)
| Script | Writes | Format | Read by |
|---|---|---|---|
| `calibration/camera_intrinsics.py` | `camera_intrinsics.yaml` | `camera_matrix` 3x3, `distortion` list, `lens_model`, `image_size [W, H]` | `camera.load_lens_calibration` |
| `calibration/board_corners.py` (click a1, h1, h8, a8 outer corners on the undistorted image of the **empty** board) | `board.yaml` | `photo_to_top_down` 3x3, `pixels_per_square`, `corner_pixels` 4x2 | `board_view.load_board_calibration` |
| same | `empty_board.png` | top-down `(N,N,3)` BGR PNG of the empty board | `main` (`cv2.imread`), passed to `occupancy.square_difference_scores` |
| `calibration/arm_board.py --park` (teleop to a safe pose out of camera view) | `park_pose.yaml` | `{joint: deg}` for all 6 joints | `kinematics.load_arm_calibration` |
| `calibration/arm_board.py --squares a1 h1 h8 a8 d4 ...` (teleop the TCP onto each square center at z=0) | `arm_board.yaml` | `board_to_robot_transform` 4x4; `samples` list of `{square, point_from_robot}`; `square_corrections` list of `{board_x_y, correction_x_y}` = leftover error after the fit | `kinematics.load_arm_calibration` |

`calibration/camera_exposure.py` (interactive, needs the camera) is the one script that writes into `config/`: its Save button replaces only the `exposure:` and `white_balance:` values in `config/hardware.yaml` and prints the change. It shows the live picture with sliders for both values, the share of blown-out and crushed pixels, and an Auto button that lets the camera choose and then locks what it chose. It reads the allowed ranges from `v4l2-ctl --list-ctrls` and prints any other automatic camera settings it finds.

Command-line options of the two vision calibration scripts (both default to writing `calibration/data/`; `--output-folder` sends the files elsewhere, e.g. a trial folder under `data/images/`):
- `camera_intrinsics.py --images DIR [--inner-corners COLS ROWS] [--square-mm MM] [--output-folder DIR]`. The checkerboard defaults and `lens_model` come from `config/vision.yaml`.
- `camera_intrinsics.py --capture DIR [--inner-corners COLS ROWS]` (needs the camera) takes the checkerboard photos automatically: it saves a full-size raw photo whenever the whole board is found, has been held still, and differs from every photo saved so far by position or tilt (corner distance) or by looking nearer or farther (size change). Settings in `vision.yaml: checkerboard.capture`. The window shows the outlines of the saved photos so the gaps are easy to see. It stops at `target_photos` or when the window is closed.
- `camera_intrinsics.py --make-checkerboard OUT.png --screen-px WIDTH HEIGHT [--inner-corners COLS ROWS]` only writes a checkerboard picture sized for a screen (no calibration). Show it full screen at 100% zoom on a laptop or phone, measure one square with a ruler, and put that in `vision.yaml: checkerboard.square_mm`.
- `board_corners.py [--photo RAW_PHOTO] [--lens LENS_YAML] [--no-lens] [--corners U V U V U V U V | --keep-corners] [--output-folder DIR]`. `--keep-corners` reuses the corners in the existing `board.yaml` and only retakes `empty_board.png` (after an exposure, white balance or lighting change; the board and camera must not have moved). It still shows the preview and asks before saving. `--no-lens` uses the photo as it is, without undistorting (for trying the clicks before the lens is calibrated); the result only fits raw photos, so with `--no-lens` it refuses to write to `calibration/data/` and needs `--output-folder`. Without `--photo` it takes one photo with the camera. Without `--corners` it opens a matplotlib window to click a1, h1, h8, a8 on the undistorted photo, then shows the top-down view with a1 labelled and saves only after you confirm. `--corners` gives the four undistorted pixel positions directly (no window).

You can extend the sparse square corrections by listing more squares on the `--squares` command. Interpolation uses all of them automatically.
The lerobot motor calibration (`~/.cache/huggingface/lerobot/calibration/robots/so_follower/HAM_follower_001.json`) is owned by lerobot and only referenced through `hardware.yaml: arm.id`.
The URDF is an external file, like the lerobot clone, referenced through `arm.yaml: urdf_path`. It is not vendored.

## 5. How to test each boundary
| Boundary | Check |
|---|---|
| `camera.take_photo/undistort` | `tools/vision_live.py --save DIR` writes the undistorted frame: straight board edges should look straight. |
| `board_view.top_down_view` | `tools/vision_live.py` draws the 8x8 grid on the top-down image: grid lines should fall on the square edges, with a1 at the bottom-left. |
| `occupancy.*` | `tools/vision_live.py --images data/images/` shows a score heatmap and occupancy for saved start/mid-game photos. Visual check. The first samples are `data/test_images/` (hand-held, 1280x720, start position, white at the bottom, **mirrored left to right**, no empty-board photo). |
| `engine.*` | pytest smoke test (skipped if the binary is missing): `best_move` on the start position returns a legal move. |
| `move_resolver.*` | `tests/test_chess_logic.py`: quiet, capture, castling both sides, en passant, promotion → queen, ambiguous slider capture (>1), noise (`[]`), and `get_robot_move` for each. |
| `kinematics.*` | `tests/test_kinematics.py`: `joint_angles_for` → `gripper_position_for` round-trip on all 64 squares and the graveyard, `square_center_on_board` corners, `None` for unreachable targets. Skipped (`pytest.importorskip("placo")`) if placo or the URDF is missing. |
| `arm` / `motion` | `python main.py --mock` logs the steps; `tests/hardware/pick_place.py e2 e4 --mock` prints the joint trajectory. Real arm only by hand. |
| `main` | `python main.py --mock --images data/images/game1/` replays a recorded game. |

`tools/collect_states.py` (labelled photo sets for milestone 2; photos in `data/images/states/<set>/`, gitignored):
- `collect_states.py empty|random|game FOLDER [--count N] [--seed S]` needs the camera and calibration. A window shows the position to set up (diagram; the game shows the move to make) next to the live top-down view, where red boxes mark squares the vision currently disagrees with. Space saves the raw photo as `NNN.jpg` and moves on; closing the window stops (everything saved so far is kept). It refuses a FOLDER that already has a `labels.yaml`.
- `collect_states.py check FOLDER` (no camera) runs the vision on every saved photo and prints the wrong squares per photo at the current `threshold`, the highest empty-square score, the lowest occupied-square score and the midpoint between them; for a game set it also checks that `find_human_move` returns each recorded move (several candidates count as found if the real move is among them).
- `labels.yaml`: `{"kind": "empty"|"random"|"game", "photos": [{"file": "000.jpg", "fen": <python-chess board_fen, the position on the board in the photo>}, ...]}`; a game set adds `"start_fen"` (full FEN) and `"moves"` (UCI strings), where photo i+1 is the board after `moves[i]`.

`tools/vision_live.py` options: no `--images` = live camera (refreshes until the window is closed); `--images DIR` = each saved photo in turn; `--calibration-folder DIR` (default `calibration/data/`; reads `camera_intrinsics.yaml`, `board.yaml`, `empty_board.png`); `--no-lens` (skip undistorting, to pair with `board_corners.py --no-lens`); `--save DIR` (with `--images`: write `<name>_straight.jpg` and `<name>_view.png` to DIR instead of opening a window; live: pressing `s` saves the raw photo to DIR, which is also how the checkerboard and board photos are taken). `board.yaml` is optional, so `--no-lens --save DIR` works before any calibration. It shows three panels: the straightened photo with the board outline, the top-down view with the 8x8 grid and occupied squares marked, and the 8x8 score heatmap (rank 8 on top). Without `empty_board.png` it shows only the first two. A **threshold slider** under the panels sets the occupied/empty line live (green rings on the top-down view, green boxes on the heatmap); it starts at the saved `threshold`, or mid-range if that is null. Its **Save threshold** button replaces only the `threshold:` line in `config/vision.yaml` (the second script allowed to write into `config/`, like `camera_exposure.py`). With `--images --save DIR` there is no window and the saved `threshold` is used.

## 6. Decisions log
- Approved: `Arm` and `MockArm` are custom classes crossing a boundary.
- Approved: variable tool pitch (§2.9). CLAUDE.md is updated.
- Approved: configs split into `vision.yaml` and `game.yaml`.
- Approved: lerobot `RobotKinematics` for IK/FK, custom IK deferred. This needs `placo` (pending your install decision).
- `find_human_move` returns a list (ambiguous slider captures). Approved: the human is shown the candidate moves and types one. Auto-resolving with per-square change scores was removed from the contract (signature, `occupancy.change_scores`, config keys); it is in OPTIONAL_IMPROVEMENTS.md and needs a contract change to add back.
- Occupancy is compared against an empty-board reference photo.
- Approved: chess_logic contracts (engine, move_resolver).
- Approved: `engine.GameOverError(ValueError)` crosses the boundary, so `main` can catch game over separately and show the result. Engine failures propagate; restart-once is in OPTIONAL_IMPROVEMENTS.
- Approved: `move_resolver` functions renamed for readability: `expected_occupancy` → `expected_board_state`, `match_move` → `find_human_move`, `ops_for` → `get_robot_move`. Behaviour unchanged.
- Approved: the robot's special moves (castling, en passant, promotion) are made by the human when prompted; the arm does quiet moves and captures (§2.13). `get_robot_move` is unchanged.
- Workspace violations raise in `plan_waypoints`. `Arm.set_joint_targets` still clamps joints as the last line of defense.
- Approved: plain-language names (CLAUDE.md rule 13) for vision, control and main: e.g. `grab` → `take_photo`, `warp` → `top_down_view`, `ik`/`fk` → `joint_angles_for`/`gripper_position_for`, `plan_ops`/`execute` → `plan_waypoints`/`run_waypoints`, `ctx` → `settings`, `acal` → `arm_calibration`. Motion's outputs are called waypoints, so "steps" means only `get_robot_move`'s output. Types and behaviour unchanged.
- Approved: exposure and white balance are locked with `v4l2-ctl` (OpenCV's V4L2 exposure properties are unreliable). `open_camera` refuses to run on auto values.
- Approved: `calibration/camera_exposure.py` writes `exposure` and `white_balance` straight into `config/hardware.yaml` (option A), so `open_camera` and its contract stay unchanged.
- Approved: `tools/vision_live.py` has a threshold slider whose Save button writes `threshold` into `config/vision.yaml`. It gives a first guess; the final value is checked on the `collect_states.py` photos.
- Approved: matplotlib (not OpenCV windows, since the env has `opencv-python-headless`) for the corner clicks and `tools/vision_live.py`.
- Synthetic vision data (`tools/make_synthetic.py`) was tried and dropped (user, 2026-10-31): vision is developed on real photos only.
- Approved: the lens checkerboard is shown on a laptop or phone screen (no printer), from `camera_intrinsics.py --make-checkerboard`.
- Deferred items with reminders are in `docs/REMINDERS.md`.
