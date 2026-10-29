# Optional improvements (not in the 2-day scope)
Each entry lists where the extension hooks in, so it can be added without changing interfaces unless noted.

## Game setup
- **Robot plays black / choose colour at startup.** Set `robot_color` in `config/game.yaml`. All physical calibration uses the "robot-white" frame. For black, `main` maps chess squares to physical squares with `sq ^ 63` (and `np.rot90(occ, 2)`). Only `main` changes.
- **Choose who moves first.** Set `robot_moves_first` in `config/game.yaml`. Only `main` changes.
- **Start from a FEN.** Set `start_fen` in `config/game.yaml`. `main` builds `chess.Board(fen)`, and the startup check compares against `expected_occupancy` of that board.

## Promotion
- **Robot promotes with a real queen.** Add a reserve spot (`reserve_xy_m` in board.yaml, board frame) and a new op `("place_from", reserve_idx, sq)`. That changes the ops contract, so update INTERFACES.md first. For now, the human swaps the piece in.
- **Human under-promotion.** Today `match_move` assumes a queen. `main` could ask on the keyboard when a pawn reaches the last rank.

## Captures
- **Multiple graveyard slots.** `graveyard_xy_m` is already a list. `motion.plan_ops` would take a slot index (or a capture count) to fill the slots in order.
- **Fewer ambiguous captures.** Use the change in a patch's appearance on the destination square (piece swapped, still occupied) to choose between candidate slider captures, instead of asking the human to type the move.

## Kinematics
- **Custom analytic IK.** Replace lerobot `RobotKinematics` (placo, iterative) with a closed-form 5-DOF solver: pan = atan2(y, x), then 2-link planar IK with the wrist set by the pitch. It needs no placo and is exact and fast. Link lengths come from the URDF. Keep the same `ik`/`fk` signatures, so only `kinematics.py` changes.

## Calibration and accuracy
- **Dense per-square offsets.** Run `calibration/arm_board.py --squares` with more (or all 64) squares. Interpolation picks them up automatically.
- **Per-piece gripper settings.** If one torque limit doesn't suit every piece, make `gripper_torque` a dict keyed by piece name.
- **Auto-detect the human's move** (no spacebar). Needs hand/arm-in-view detection plus a stable-frame check.
