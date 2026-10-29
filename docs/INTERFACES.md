# Interfaces
Source of truth for what data moves between modules. Written/updated BEFORE code changes.
Status: DRAFT v2 (lerobot IK, split configs), awaiting review.

## 1. Data flow (one line per hop)
Only `main.py` calls across modules. `motion.py` is the one module that calls others (`kinematics`, `arm`), as CLAUDE.md allows.

```
startup
  main -> camera.load_intrinsics(path)                  -> intr dict
  main -> board_view.load_board_calib(path)             -> bcal dict
  main -> kinematics.load_arm_calib(dir, arm_cfg, mock) -> acal dict
  main -> engine.open_engine(engine_cfg)                -> chess.engine.SimpleEngine
  main -> camera.open_camera(camera_cfg)                -> cv2.VideoCapture
  main -> Arm(hw_cfg, arm_cfg) | MockArm(...)  .connect()
  main -> motion.park(arm, ctx)

perceive (used after every move, human or robot)
  main -> camera.grab(cap)                              -> raw frame      (H,W,3) uint8 BGR
  main -> camera.undistort(raw, intr)                   -> undistorted    (H,W,3) uint8 BGR
  main -> board_view.warp(undistorted, bcal)            -> top            (N,N,3) uint8 BGR
  main -> occupancy.square_scores(top, empty_top, vcfg) -> scores         (8,8) float32
  main -> occupancy.occupancy(scores, vcfg)             -> occ            (8,8) bool

human turn (after spacebar)
  main -> occupancy.change_scores(top_before, top_after, vcfg) -> change   (8,8) float32   [only if ambiguous_capture: auto; not built yet]
  main -> move_resolver.match_move(board, before, after, change, margin) -> list[chess.Move]  (0, 1 or many)

robot turn
  main -> engine.best_move(eng, board, engine_cfg)      -> chess.Move
  if move is special (§2.13): main prints it, the human makes it on the board, presses spacebar -> go to verify
  else:
  main -> move_resolver.ops_for(board, move)            -> list[op tuple]
  main -> motion.plan_ops(ops, heights, ctx)            -> list[step tuple]
          motion -> kinematics.square_xy / board_to_base / tool_pitch / ik
  main -> motion.execute(arm, steps, ctx)               -> None
          motion -> arm.read() / arm.send() / arm.set_gripper_torque()
  main -> motion.park(arm, ctx)                         -> None   (arm out of camera view)

verify (after robot move, and at startup)
  main -> move_resolver.expected_occupancy(board)       -> (8,8) bool; main compares with perceived occ

abort (Ctrl-C anywhere)
  main -> motion.park(arm, ctx) -> arm.torque_off() -> arm.disconnect()   (second Ctrl-C: torque_off at once)
```

## 2. Conventions (defined once, used everywhere)

### 2.1 Game orientation
- **Robot plays white and moves first** (`config/game.yaml: robot_color: white`). For now, any other value raises `NotImplementedError` at startup.
- Robot sits at the rank-1 edge, centered. Human sits opposite, at the rank-8 edge.
- **Physical frame = "robot-white" frame.** Every physical calibration (corner clicks, teleop squares) uses square names as if the robot plays white. To support black later, `main` rotates the 8x8 grid 180° (`sq ^ 63`) between chess squares and physical squares. Nothing else changes. See `docs/OPTIONAL_IMPROVEMENTS.md`.

### 2.2 Squares
- python-chess square ints, 0-63: `sq = rank_idx*8 + file_idx`, with `file_idx` 0=a..7=h and `rank_idx` 0=rank 1..7=rank 8. So a1=0, h1=7, e2=12, e4=28, a8=56, h8=63.
- Square names appear only in user-facing text and the CLI (`chess.square_name`, `chess.parse_square`).

### 2.3 Occupancy array
- `np.ndarray`, shape `(8, 8)`, dtype `bool`, indexed `occ[rank_idx, file_idx]`. `occ[0, 0]` is a1 and `occ[7, 7]` is h8.
- Therefore `occ.reshape(64)[sq]` is the occupancy of python-chess square `sq`.
- `True` = occupied. Scores (`float32`, same layout) are higher when the square is more likely occupied.
- This is **not** image layout. To display it with rank 8 on top, use `np.flipud(occ)`.

### 2.4 Image pixel frame (raw and undistorted)
- `np.ndarray (H, W, 3) uint8` in BGR (OpenCV) order, indexed `[v, u]`. `u` runs right and `v` runs down, origin at the top-left pixel, units px.
- Raw and undistorted frames have the same size `(H, W)` = camera resolution from `config/hardware.yaml`.

### 2.5 Top-down board image
- `np.ndarray (N, N, 3) uint8` BGR, where `N = 8 * px_per_square`. `px_per_square` is stored in `calibration/data/board.yaml`, copied there from `config/vision.yaml` when the calibration was made.
- It covers exactly the 8x8 playing area, with no border. **a1 is at the bottom-left and h8 at the top-right**, so the image looks like the board seen from the robot's (white's) side.
- The center of square (file_idx f, rank_idx r) is at `u = (f + 0.5)*p` and `v = (7 - r + 0.5)*p`, with `p = px_per_square`.
- It is a homography of the **board plane** (z = 0). Tall pieces lean outward, so the occupancy check samples only a centered patch at each square center.

### 2.6 Board plane frame (meters)
- Origin at the **outer corner of a1** of the 26.5 cm playing area. +x runs along rank 1 toward the h-file, +y along the a-file toward rank 8, +z points up, and z = 0 is the board surface.
- Square center: `x = (f + 0.5)*s`, `y = (r + 0.5)*s`, with `s = square_size_m` = 0.265/8 = 0.033125 m. Example: e2 is at (0.1490625, 0.0496875).
- Off-board spots (graveyard, spare pieces) are also given in this frame, e.g. `x < 0` or `y < 0`.

### 2.7 Robot base frame (meters)
- This is the **world frame of lerobot `RobotKinematics`**, which is the `base` link of the SO101 URDF `so101_new_calib.urdf` (SO-ARM100 repo). Kinematics uses lerobot's IK/FK (placo). A custom IK is in `docs/OPTIONAL_IMPROVEMENTS.md`.
- Axis directions and origin: **TODO, documented here once the URDF is downloaded**. Expected: origin at the base mount, +z up. Which horizontal axis points at the board is read from the URDF.
- With the robot playing white, **a1 is front-left of the robot and h1 is front-right**.
- `T_board_base` (4x4, float64, homogeneous) maps board-frame points to base-frame points. It is fitted by `calibration/arm_board.py`, so a rough setup is fine and the axis convention doesn't affect correctness.

### 2.8 Joint angles
- Joint dicts at every boundary are `dict[str, float]` with bare lerobot names: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`. `arm.py` adds and strips the `.pos` suffix.
- The 5 body joints are in **lerobot SO101 degrees** (`use_degrees=True`). 0° is the middle of the range recorded by `lerobot-calibrate` (calibration id `HAM_follower_001`).
- `so101_new_calib.urdf` is built for this convention (URDF joint zero = lerobot 0°). So **lerobot degrees go straight into `RobotKinematics`** with no zero offsets. Any leftover mismatch shows up as residuals in `arm_board.py` and is absorbed by the per-square offsets. Check it on the bench (see REMINDERS).
- `gripper` is lerobot `RANGE_0_100`. Which end is open is TODO, to be checked on the bench. It is never passed to IK.

### 2.9 Tool pitch
- Degrees from horizontal: positive points downward, 90 = straight down.
- It is not solved for. It follows a config schedule linear in horizontal reach `ρ = hypot(x, y)` in the base frame: `pitch_near_deg` at `ρ_near_m` and `pitch_far_deg` at `ρ_far_m`, clamped outside that range. With both values equal, this becomes a constant.
- `wrist_roll` is a fixed config constant (`wrist_roll_deg`).
- Inside `kinematics.ik`, the desired 4x4 TCP pose is built from the position, yaw `atan2(y, x)` (the 5-DOF arm can't choose its yaw), the pitch, and the fixed roll. lerobot's IK then solves it with orientation weight `ik_orientation_weight`. The result is accepted only if FK reproduces it within `ik_pos_tol_m` and `ik_pitch_tol_deg`.

### 2.10 Other units
- Lengths are in meters. Times are in seconds. Angles at boundaries are in degrees.
- The gripper torque limit is in raw Feetech `Max_Torque_Limit` units, 0-1000. lerobot's default for the gripper is 500.

### 2.11 Ops and steps
- Op (chess_logic -> control): `("move", from_sq, to_sq)` or `("remove", sq)`. Squares are python-chess ints.
- Step (motion internal, public for debugging): `("joints", dict[str, float])` (5 body joints, degrees) or `("gripper", "open" | "close")`.

### 2.12 Config dicts
`main` loads each YAML once and passes plain dicts in. Modules never read `config/` themselves.
Files: `hardware.yaml` (camera, arm port and id), `arm.yaml` (kinematics, tool, motion, gripper, safety), `board.yaml` (board, pieces, off-board spots), `vision.yaml` (warp and occupancy tuning), `game.yaml` (colour, first move, start FEN, Stockfish).
`ctx` (motion) = `{"arm": arm.yaml dict, "board": board.yaml dict, "acal": arm calib dict}`.

### 2.13 Who moves the robot's pieces
- The **arm** executes the robot's normal moves: quiet moves and ordinary captures (captured piece to the graveyard, then the move).
- **Special moves are made physically by the human** when prompted: castling, en passant and promotion. `main` decides with python-chess: `board.is_castling(m) or board.is_en_passant(m) or m.promotion is not None`. It prints the move (SAN and UCI; for promotion, "replace the pawn with a queen"), waits for spacebar, then runs the normal verify step against `expected_occupancy`.
- `ops_for` still returns full ops for special moves (unchanged contract and tests), so handing them to the arm later only changes `main`. See `docs/OPTIONAL_IMPROVEMENTS.md`.

## 3. Function contracts

### vision/camera.py
| | |
|---|---|
| `open_camera(camera_cfg: dict) -> cv2.VideoCapture` | Opens `camera_cfg["device"]` (a `/dev/v4l/by-id/...` path) at `width`x`height` and locks exposure and white balance from config. **Raises `RuntimeError`** if the camera can't be opened or the resolution is refused. |
| `grab(cap: cv2.VideoCapture, flush: int) -> np.ndarray` | Reads and discards `flush` buffered frames, then returns a fresh `(H,W,3) uint8 BGR` raw frame. **Raises `RuntimeError`** if the read fails. |
| `load_intrinsics(path: str) -> dict` | Returns `{"K": (3,3) float64, "D": (4,) or (5,) float64, "model": "fisheye" or "pinhole", "size": (W, H) int}`. **Raises `FileNotFoundError`**. |
| `undistort(img: np.ndarray, intr: dict) -> np.ndarray` | Takes a raw frame and returns an undistorted frame with the same shape and dtype (`balance`/`alpha` = 0, so pixels outside the original are black). **Raises `ValueError`** if `img` size != `intr["size"]`. |

Example: `undistort(grab(cap, 2), intr)` gives `(1080, 1920, 3) uint8`.
Offline: use `cv2.imread("data/images/x.png")` instead of `grab`.

### vision/board_view.py
| | |
|---|---|
| `load_board_calib(path: str) -> dict` | Returns `{"H": (3,3) float64 (undistorted px -> top px), "px_per_square": int, "corners_px": (4,2) float64 [a1, h1, h8, a8] outer corners in undistorted px}`. **Raises `FileNotFoundError`**. |
| `warp(img: np.ndarray, bcal: dict) -> np.ndarray` | Takes an undistorted `(H,W,3) uint8` image and returns top-down `(N,N,3) uint8` (§2.5). Never fails for a valid array. |

Example: with `px_per_square = 50`, `warp(und, bcal).shape == (400, 400, 3)`, and the e2 center is at `[v=325, u=225]`.

### vision/occupancy.py
| | |
|---|---|
| `square_scores(top: np.ndarray, empty_top: np.ndarray, vision_cfg: dict) -> np.ndarray` | Compares a centered patch (`patch_frac` of the square side) of each square in `top` against the same patch of the empty-board reference `empty_top` (same shape). Returns `(8,8) float32`, ≥ 0, layout §2.3. **Raises `ValueError`** on shape mismatch. |
| `occupancy(scores: np.ndarray, vision_cfg: dict) -> np.ndarray` | Returns `scores > vision_cfg["threshold"]` as `(8,8) bool`. |
| `change_scores(top_before: np.ndarray, top_after: np.ndarray, vision_cfg: dict) -> np.ndarray` | **Not built yet. Only used when `game.yaml: ambiguous_capture: auto`** (see OPTIONAL_IMPROVEMENTS.md). Compares the same centered patch of each square between two top-down images of the same game. Returns `(8,8) float32`, ≥ 0, layout §2.3; higher means the square looks more different. This is used to resolve ambiguous captures, where the target square stays occupied but the piece swaps colour. It never labels a colour. **Raises `ValueError`** on shape mismatch. |

Example: for the start position, `occupancy(...)` has rows 0, 1, 6 and 7 all True.
The method compares against an empty-board reference image, so it doesn't rely on piece colour. The scoring internals are an implementation detail.

### chess_logic/engine.py
| | |
|---|---|
| `open_engine(engine_cfg: dict) -> chess.engine.SimpleEngine` | Starts the binary at `engine_cfg["path"]` and sets `Skill Level` (0-20). **Raises `FileNotFoundError`** if the binary is missing, and `chess.engine.EngineError` if it fails to start. |
| `best_move(eng, board: chess.Board, engine_cfg: dict) -> chess.Move` | Thinks for `think_time_s` (time limit only). Does not modify `board`. **Raises `engine.GameOverError`** (subclass of `ValueError`) if `board.is_game_over()` (no draw claims: threefold/50-move do not count). If the engine crashes or fails, its `chess.engine.EngineError` / `EngineTerminatedError` **propagates** (no restart); `main` stops explicitly. |
| `close_engine(eng) -> None` | Quits the engine. Safe to call twice. |

Example: `best_move(eng, chess.Board(), cfg)` might return `Move.from_uci("e2e4")`.

### chess_logic/move_resolver.py
| | |
|---|---|
| `expected_occupancy(board: chess.Board) -> np.ndarray` | `(8,8) bool`: True where python-chess has a piece. Never fails. |
| `match_move(board: chess.Board, before: np.ndarray, after: np.ndarray, change: np.ndarray \| None = None, margin: float = 0.0) -> list[chess.Move]` | Finds the human's move. `board` is the position before the move (unchanged). `before` and `after` are perceived `(8,8) bool`. Step 1: collect every legal move `m` for which the squares changed by `m` (from `expected_occupancy`) equal `before ^ after`. Promotions to the same square are merged into the **queen** promotion. Step 2, only if more than one candidate remains and `change` (`(8,8) float32` from `occupancy.change_scores`) is given: score each candidate by `change` at its `to_square`, and return just the best one if it beats the runner-up by more than `margin` (`config/vision.yaml: ambiguity_margin`, passed in by `main`). **Never raises.** `[]` means no match, 1 element means resolved, and more than one means still ambiguous; `main` then asks the human (see main.py below). With `ambiguous_capture: ask` (current default), `main` passes `change=None`, so every candidate is returned. |
| `ops_for(board: chess.Board, move: chess.Move) -> list[tuple]` | `board` is the position **before** `move` (unchanged). Returns ops (§2.11), with removals first. **Raises `ValueError`** if `move` is illegal. |

Comparing changed squares (`before ^ after`) instead of the whole grid means that a square vision consistently misreads doesn't break matching.

`match_move` example: with a white queen on d1 that can take on d7 or g4, occupancy shows only that d1 emptied, so both moves are candidates. With `change[g4] = 40.0`, `change[d7] = 2.0` and `margin = 10.0`, it returns `[Move.from_uci("d1g4")]`. With no `change` (the current `ask` mode), it returns both.

`ops_for` examples:
- Quiet move e2e4: `[("move", 12, 28)]`
- Capture e4xd5: `[("remove", 35), ("move", 28, 35)]`
- White castling O-O: `[("move", 4, 6), ("move", 7, 5)]` (king first, then rook)
- En passant e5xd6: `[("remove", 35), ("move", 36, 43)]`
- Promotion e7e8=Q: `[("move", 52, 60)]`.
- Castling, en passant and promotion are special moves (§2.13): today `main` has the human make them and does not send these ops to `motion`.

### control/kinematics.py (config dicts + `lerobot.model.kinematics.RobotKinematics`)
| | |
|---|---|
| `load_arm_calib(data_dir: str, arm_cfg: dict, allow_nominal: bool) -> dict` | Returns `{"kin": RobotKinematics (built from arm_cfg["urdf_path"], target_frame), "T_board_base": (4,4) float64, "offsets": list[((x, y), (dx, dy))] (board m), "park": dict[str, float] deg}`. If calibration files are missing: with `allow_nominal` (only `--mock`) it uses `arm_cfg["nominal"]`; otherwise it **raises `FileNotFoundError`**. A missing URDF always raises `FileNotFoundError`. If placo is missing, lerobot's `ImportError` propagates. |
| `square_xy(sq: int, board_cfg: dict) -> tuple[float, float]` | Square center in the board frame, in meters (§2.6). **Raises `ValueError`** if `sq` is not 0-63. |
| `board_to_base(xy: tuple[float, float], z: float, acal: dict) -> np.ndarray` | Board-frame `(x, y)` plus height `z` above the board gives `(3,) float64` in the base frame. It applies `T_board_base` plus the xy offset interpolated from `acal["offsets"]` (inverse-distance weighting; zero if there are no samples). |
| `tool_pitch(p_base: np.ndarray, arm_cfg: dict) -> float` | Degrees from horizontal (§2.9). |
| `ik(p_base: np.ndarray, pitch_deg: float, seed: dict[str, float], arm_cfg: dict, acal: dict) -> dict[str, float] \| None` | TCP position (base frame, m) plus pitch gives the 5 body joints in degrees (§2.8). `seed` is the starting guess for the iterative solver: the current joints, or the previous waypoint. **Returns `None`** if the FK check fails (unreachable) or any joint is outside `joint_limits_deg`. |
| `fk(joints: dict[str, float], arm_cfg: dict, acal: dict) -> tuple[np.ndarray, float]` | 5 body joints in degrees give the TCP `(3,) float64` in base m (target frame plus `tcp_offset_m`) and the pitch in degrees. Missing keys **raise `KeyError`**. |

Example: `p = board_to_base(square_xy(12, bcfg), 0.02, acal)`, then `fk(ik(p, tool_pitch(p, acfg), acal["park"], acfg, acal), acfg, acal)` gives back `≈ (p, pitch)` to within `ik_pos_tol_m` and `ik_pitch_tol_deg`.

### control/arm.py (custom classes crossing a boundary: approved)
`Arm(hw_cfg: dict, arm_cfg: dict)` wraps lerobot `SO101Follower`. `MockArm(hw_cfg, arm_cfg, start: dict[str, float])` has the same methods, so the two can be swapped.
| | |
|---|---|
| `connect() -> None` | **Raises `ConnectionError`** if the port is missing (from lerobot). `MockArm` never fails. |
| `read() -> dict[str, float]` | All 6 joints (§2.8). |
| `send(target: dict[str, float]) -> dict[str, float]` | Takes any subset of the 6 joints. Each value is **clamped** to `joint_limits_deg` / gripper 0-100, and the dict actually sent is returned. Never blocks for motion; `motion` interpolates. |
| `set_gripper_torque(limit: int) -> None` | Writes the gripper `Max_Torque_Limit` (0-1000). When the gripper is commanded closed, it stalls on the piece at this force instead of needing a per-piece position. |
| `torque_off() -> None` | Disables torque on all motors. Never raises (errors are logged). |
| `disconnect() -> None` | Safe to call twice. |

### control/motion.py
| | |
|---|---|
| `plan_ops(ops: list[tuple], heights: dict[int, float], ctx: dict) -> list[tuple]` | `heights[sq]` is the height (m) of the piece standing on `sq` **before** the move, for every square an op picks from. Returns steps (§2.11) covering the whole move: for each pick, hover, open, descend to `grip_height_frac*h`, close, lift to travel height; then above the destination, descend, open, lift. A `remove` goes to `graveyard_xy_m[0]`. Plans everything before anything moves. **Raises `ValueError`** if any target is outside the board workspace (+ `workspace_margin_m`, graveyard allowed), below `min_z_m`, or unreachable by `ik`. |
| `execute(arm, steps: list[tuple], ctx: dict) -> None` | Interpolates in joint space from `arm.read()` toward each step, at `max_joint_speed_deg_s * max_speed_scale` and `control_hz`. Small ticks keep Ctrl-C responsive. **Raises `RuntimeError`** if the arm does not reach a step within `settle_tol_deg` and `step_timeout_s`. |
| `park(arm, ctx) -> None` | Moves to `acal["park"]` (via travel height) the same way `execute` does. |

Example: `plan_ops([("move", 12, 28)], {12: 0.028}, ctx)` gives about 10 steps, starting with `("gripper", "open")` and a `("joints", {...})` hover above e2.

### main.py (not imported by anyone)
`main(argv: list[str] | None = None) -> int`, with flags `--mock` and `--images DIR` (offline frames). Exit code 0 means the game ended normally, 1 means aborted.
**Ambiguity prompt** (`match_move` returns more than one move; always, with `ambiguous_capture: ask`). `main` builds it from python-chess (colour, piece name, squares):
```
I couldn't tell which capture you made.
Your black queen on d8 moved. It could have gone to:
  1. d8 → d2   capturing white pawn on d2
  2. d8 → h4   capturing white knight on h4
Type the move (e.g. d8h4) or just the destination square (e.g. h4):
```
Non-capture candidates have no "capturing" part. It accepts the full UCI move, the destination square, or the list number. Anything else prints "That isn't one of the options above" and asks again.

`heights` is built in `main` from python-chess: `{sq: board_cfg["piece_height_m"][chess.piece_name(p.piece_type)] for sq, p in board.piece_map().items()}`.

## 4. Calibration artifacts (`calibration/data/`, generated only)
| Script | Writes | Format | Read by |
|---|---|---|---|
| `calibration/camera_intrinsics.py` | `camera_intrinsics.yaml` | `K` 3x3, `D` list, `model`, `size [W, H]` | `camera.load_intrinsics` |
| `calibration/board_corners.py` (click a1, h1, h8, a8 outer corners on the undistorted image of the **empty** board) | `board.yaml` | `H` 3x3, `px_per_square`, `corners_px` 4x2 | `board_view.load_board_calib` |
| same | `empty_board.png` | top-down `(N,N,3)` BGR PNG of the empty board | `main` (`cv2.imread`), passed to `occupancy.square_scores` |
| `calibration/arm_board.py --park` (teleop to a safe pose out of camera view) | `park_pose.yaml` | `{joint: deg}` for all 6 joints | `kinematics.load_arm_calib` |
| `calibration/arm_board.py --squares a1 h1 h8 a8 d4 ...` (teleop the TCP onto each square center at z=0) | `arm_board.yaml` | `T_board_base` 4x4; `samples` list of `{sq, xyz_base}`; `offsets` list of `{xy, dxy}` = residuals after the fit | `kinematics.load_arm_calib` |

You can extend the sparse offsets by listing more squares on the `--squares` command. Interpolation uses all of them automatically.
The lerobot motor calibration (`~/.cache/huggingface/lerobot/calibration/robots/so_follower/HAM_follower_001.json`) is owned by lerobot and only referenced through `hardware.yaml: arm.id`.
The URDF is an external file, like the lerobot clone, referenced through `arm.yaml: urdf_path`. It is not vendored.

## 5. How to test each boundary
| Boundary | Check |
|---|---|
| `camera.grab/undistort` | `tools/vision_live.py --save` writes the undistorted frame: straight board edges should look straight. |
| `board_view.warp` | `tools/vision_live.py` draws the 8x8 grid on the top-down image: grid lines should fall on the square edges, with a1 at the bottom-left. |
| `occupancy.*` | `tools/vision_live.py --images data/images/` shows a score heatmap and occupancy for saved start/mid-game photos. Visual check. The first samples are `data/test_images/` (hand-held, 1280x720, start position, white at the bottom). |
| `engine.*` | pytest smoke test (skipped if the binary is missing): `best_move` on the start position returns a legal move. |
| `move_resolver.*` | `tests/test_chess_logic.py`: quiet, capture, castling both sides, en passant, promotion → queen, ambiguous slider capture (>1), noise (`[]`), and `ops_for` for each. |
| `kinematics.*` | `tests/test_kinematics.py`: IK→FK round-trip on all 64 squares and the graveyard, `square_xy` corners, `None` for unreachable targets. Skipped (`pytest.importorskip("placo")`) if placo or the URDF is missing. |
| `arm` / `motion` | `python main.py --mock` logs the steps; `tests/hardware/pick_place.py e2 e4 --mock` prints the joint trajectory. Real arm only by hand. |
| `main` | `python main.py --mock --images data/images/game1/` replays a recorded game. |

## 6. Decisions log
- Approved: `Arm` and `MockArm` are custom classes crossing a boundary.
- Approved: variable tool pitch (§2.9). CLAUDE.md is updated.
- Approved: configs split into `vision.yaml` and `game.yaml`.
- Approved: lerobot `RobotKinematics` for IK/FK, custom IK deferred. This needs `placo` (pending your install decision).
- `match_move` returns a list (ambiguous slider captures). Approved: with `game.yaml: ambiguous_capture: ask` (current), the human is shown the candidate moves and types one. `occupancy.change_scores` + `ambiguous_capture: auto` is parked in OPTIONAL_IMPROVEMENTS.md; the contract keeps its optional arguments, so adding it later needs no interface change.
- Occupancy is compared against an empty-board reference photo.
- Approved: chess_logic contracts (engine, move_resolver).
- Approved: `engine.GameOverError(ValueError)` crosses the boundary, so `main` can catch game over separately and show the result. Engine failures propagate; restart-once is in OPTIONAL_IMPROVEMENTS.
- Approved: the robot's special moves (castling, en passant, promotion) are made by the human when prompted; the arm does quiet moves and captures (§2.13). `ops_for` is unchanged.
- Workspace violations raise in `plan_ops`. `Arm.send` still clamps joints as the last line of defense.
- Deferred items with reminders are in `docs/REMINDERS.md`.
