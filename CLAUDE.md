# CLAUDE.md: SO101_Chess

## Objective
A LeRobot SO101 arm plays chess autonomously against a human, in a 2-day build.
Pipeline: camera -> board occupancy -> (human move identified) -> Stockfish reply -> arm executes the move -> camera verifies.
Deterministic, modular. NOT end-to-end learning, NOT a VLA.

## Hardware and physical facts (treat as constraints)
- Arm: SO101 follower (5 DOF + gripper), clamped to the table. Leader arm is used only for teleop calibration.
- Gripper: stock jaws, opening is **sideways** when angled. TCP = end/middle of the open jaws. Tool pitch (from horizontal) **varies with reach**: steep near the base, shallower at the far edge. It follows a linear schedule in reach set in `config/arm.yaml` (`pitch_near_deg`/`pitch_far_deg`). Do not solve for tool pitch; read it from config.
- Camera: IMX335, USB, ~106 deg diagonal FOV (heavy barrel distortion, undistort first), mounted overhead ~60 cm above board center, exposure/white balance locked.
- Board: 30x30 cm total, 26.5x26.5 cm playing area (3.3125 cm squares), fixed to the table. Robot plays white for now, sitting at the rank-1 edge; the human sits opposite. Pieces: wood, 2 cm base diameter, up to 5 cm tall. Pieces and board are similar brown/tan colors, so vision must NOT rely on piece color. Vision answers only "occupied or empty" per square.
- Parallax: tall pieces shift in the image. Sample a small patch at each square center, in the board-plane-warped image.
- Opponent: a human. They press a key (spacebar) when their move is done. No auto-detection.
- Compute: laptop with NVIDIA GPU (not needed for the core pipeline).

## Architecture and dependency rules
Only `main.py` wires modules together. Modules must not import each other except as listed.

| Module | Job | May import |
|---|---|---|
| `vision/camera.py` | frames, locked exposure, undistort | calibration data files, config |
| `vision/board_view.py` | image -> top-down board image | camera output type, calibration data |
| `vision/occupancy.py` | top-down board -> 8x8 bool grid | board_view output type |
| `chess_logic/engine.py` | Stockfish wrapper: board -> best move | python-chess only |
| `chess_logic/move_resolver.py` | `match_move` (occupancy before/after -> human's legal move), `ops_for` (move -> list of move/remove ops: captures, castling, en passant) | python-chess, numpy only |
| `control/kinematics.py` | square/xyz -> joint angles via lerobot `RobotKinematics` (placo + SO101 URDF) | config, `lerobot.model.kinematics` |
| `control/motion.py` | ops -> waypoints/trajectory -> arm calls | kinematics, arm |
| `control/arm.py` | hardware driver + `MockArm` | lerobot |
| `calibration/*` | one-off scripts that write to `calibration/data/` | anything they need |
| `tools/`, `tests/` | debugging and tests | anything |

`vision/` must never import `control/`, and vice versa. `chess_logic/` knows nothing about cameras or motors.

## The interface contract (most important rule)
`docs/INTERFACES.md` is the source of truth for everything that crosses a file boundary.
- For every boundary give: function signature, input/output **type, shape, dtype, units, coordinate frame, value range**, one concrete example, and what happens on failure (return None vs raise, and which exception).
- Define the coordinate frames and conventions once at the top (image pixels, top-down board image, board plane in meters, robot base frame; square indexing; occupancy array layout, including which index is rank 1 and which side the robot plays; joint angle units).
- **Change the interface doc first, get my approval, then change code.** Never silently change a signature.
- Prefer plain types: python-chess `Move`/`Board`, python-chess square ints (0-63), numpy arrays, floats, tuples, dicts. No custom classes across module boundaries unless I approve.
- Ops format between chess_logic and control: `("move", from_sq, to_sq)` or `("remove", sq)`.

## Working rules
1. **Plan before coding.** For any non-trivial task, propose the plan and the files you will touch, wait for my approval, then implement.
2. **Minimal files.** The directory structure is final. Do not add files or folders without asking. Do not split a file unless it is clearly too large (~300+ lines) and tell me why first.
3. **Small steps, each verifiable.** After each step, say how I can check it (a test, a script, a saved image).
4. **Hardware safety.** Never run anything that moves the real arm unless I explicitly say so in that message. Default everything to `MockArm`. Keep speeds low (`max_speed_scale` in config), enforce joint limits, and clamp targets to the board workspace. Add a fast abort path (Ctrl-C returns the arm to a safe pose / disables torque).
5. **No magic numbers.** Anything tunable at the bench (heights, speeds, thresholds, offsets) lives in `config/*.yaml` or `calibration/data/`. Code reads it, never hardcodes it.
6. **Generated vs hand-edited.** Never hand-edit files in `calibration/data/`; regenerate them with the calibration scripts.
7. **Tests.** pytest only for `chess_logic` and `kinematics` (pure logic). Cover castling, en passant, captures, ambiguous or failed matches, and IK round-trip with forward kinematics. Vision and control are validated visually and on hardware, not with unit tests. `tests/hardware/pick_place.py` is run by hand and must not be collected by pytest.
8. **Offline-first.** Vision must be developable from saved images in `data/images/`. Control must be developable with `MockArm`. Do not make anything require the physical setup to import or test.
9. **Logging.** `main.py` logs every move with the before/after frames to `data/logs/<timestamp>/`.
10. **Environment.** Use the existing conda env `lerobot` (`conda activate lerobot`). Never create a venv or a new conda env, and never install into base. The env already provides lerobot, numpy, opencv and pyyaml. Do not `pip install` opencv-python or anything that could replace those packages. Ask before adding any new package, and before installing, check what is already present (`pip list`). Stockfish is a system binary. LeRobot comes from my local clone, not vendored.
11. **Git.** You may `git add` and `git commit` at the end of each verified step with a short descriptive message. Never `git push`, force-push, rewrite history, or change the remote unless I explicitly ask. Never commit `data/images/` or `data/logs/` (gitignored) or secrets. The repo path contains a space (`~/Projects/HAM ARC/SO101_Chess`), so always quote paths in shell commands.
12. **Ask, don't guess.** If a physical value, frame convention, or hardware behavior is unknown, ask me and mark it TODO in the config. Do not invent numbers.

## Error handling policy (in `main.py`)
After each robot move, compare the observed occupancy with what python-chess expects.
1. Match: continue.
2. Mismatch: re-read occupancy, retry the move once.
3. Still mismatch: stop, print the expected vs observed squares, ask the human to fix the board, wait for a keypress.
Failures must be explicit. Never silently continue on an unverified board.

## Don't
- Don't train models or add learned components unless I ask.
- Don't classify piece type or color in vision.
- Don't hardcode square coordinates in motion code; they come from the board-to-robot calibration plus per-square offsets.
- Don't change `config/` values to make a test pass without telling me.
- Don't write long docstrings or extra abstractions. Keep code short and readable.

## Commands
- Env: `conda activate lerobot` (always, before anything else)
- Install (once): `pip install -r requirements.txt`, which only adds python-chess and pytest. Stockfish is installed separately. This is an app run from the repo root, not an installable package: do not add a `[project]` table, `setup.py`, or `pip install -e .`.
- Tests: `PYTHONPATH= pytest` (until ROS PYTHONPATH is fixed, see `docs/REMINDERS.md`)
- Play: `python main.py` (add `--mock` to use `MockArm`)
- Debug vision: `python tools/vision_live.py`
- Hardware pick/place test (moves the arm!): `python tests/hardware/pick_place.py e2 e4`

## Docs
- `docs/INTERFACES.md`: the contract.
- `docs/REMINDERS.md`: open questions and reminders per subsystem. **Read the relevant section before starting a subsystem and raise its items with me.**
- `docs/OPTIONAL_IMPROVEMENTS.md`: deferred ideas; do not implement unless I ask.

## Build order
1. `docs/INTERFACES.md` (plan, no code)
2. `chess_logic/move_resolver.py` + tests, then `engine.py`
3. `vision/*` on saved images, then live
4. `control/kinematics.py` + tests
5. `control/arm.py` MockArm, then `motion.py` against MockArm
6. Calibration scripts, then real hardware via `tests/hardware/pick_place.py`
7. `main.py` game loop with verify/retry, first with MockArm, then real
