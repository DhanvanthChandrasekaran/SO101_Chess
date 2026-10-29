# Optional improvements (not in the 2-day scope)
Each entry lists where the extension hooks in, so it can be added without changing interfaces unless noted.

## Game setup
- **Robot plays black / choose colour at startup.** Set `robot_color` in `config/game.yaml`. All physical calibration uses the "robot-white" frame. For black, `main` maps chess squares to physical squares with `sq ^ 63` (and `np.rot90(occ, 2)`). Only `main` changes.
- **Choose who moves first.** Set `robot_moves_first` in `config/game.yaml`. Only `main` changes.
- **Start from a FEN.** Set `start_fen` in `config/game.yaml`. `main` builds `chess.Board(fen)`, and the startup check compares against `expected_board_state` of that board.

## Special moves by the arm
- **Arm castles and takes en passant.** `get_robot_move` already returns the steps (king then rook; remove the passed pawn, then move). Only `main` changes: stop sending these moves to the human and pass their steps to `motion`. Validate on hardware first (two-piece moves need clear travel paths).

## Promotion
- **Robot promotes with a real queen.** Add a reserve spot (`reserve_xy_m` in board.yaml, board frame) and a new step `("place_from", reserve_slot, square)`. That changes the steps contract, so update INTERFACES.md first. For now, the human makes the whole promotion move.
- **Human under-promotion.** Today `find_human_move` assumes a queen. `main` could ask on the keyboard when a pawn reaches the last rank.

## Captures
- **Auto-resolve ambiguous captures.** Build `occupancy.change_scores` (per-square before/after patch difference; a capture swaps piece colour on the target square, so it changes a lot). Add `change` ((8,8) float32) and `margin` (float, from a new `vision.yaml: ambiguity_margin` tuned on recorded game photos) back to `find_human_move`: when several candidates remain, return the one whose `to_square` change score beats the runner-up by more than `margin`. Contract change (INTERFACES.md first): `find_human_move` signature, a new `occupancy.change_scores` row, and the human-turn data flow. The human prompt stays as the fallback.
- **Multiple graveyard slots.** `graveyard_xy_m` is already a list. `motion.plan_waypoints` would take a slot index (or a capture count) to fill the slots in order.

## Engine
- **Restart Stockfish once on a crash.** Today `best_move` lets `EngineError`/`EngineTerminatedError` propagate and `main` stops. `main` could instead catch it, `close_engine` + `open_engine` again, and retry `best_move` once. Done in `main`, this needs no signature change.

## Kinematics
- **Custom analytic IK.** Replace lerobot `RobotKinematics` (placo, iterative) with a closed-form 5-DOF solver: pan = atan2(y, x), then 2-link planar IK with the wrist set by the pitch. It needs no placo and is exact and fast. Link lengths come from the URDF. Keep the same `joint_angles_for`/`gripper_position_for` signatures, so only `kinematics.py` changes.

## Calibration and accuracy
- **Dense per-square offsets.** Run `calibration/arm_board.py --squares` with more (or all 64) squares. Interpolation picks them up automatically.
- **Per-piece gripper settings.** If one torque limit doesn't suit every piece, make `gripper_torque` a dict keyed by piece name.
- **Auto-detect the human's move** (no spacebar). Needs hand/arm-in-view detection plus a stable-frame check.
