"""Move the limp follower arm by hand so it learns where the board is; also records the resting pose and measures spots.

Usage: python calibration/arm_board.py [--squares a1 h1 h8 a8 ...] [--output-folder DIR]
       python calibration/arm_board.py --park | --point | --refit
The arm's motors stay switched OFF the whole time, so the arm goes limp: hold it when it starts.
Board: hold a pawn in the jaws (jaw tips at about grip_height_frac of the pawn's height), stand it on the centre
of each square in the pose you would pick a piece with, and press Enter. Writes arm_board.yaml and prints
suggested tilt and wrist-roll values for config/arm.yaml.
--park: move the arm to its resting pose (out of the camera's view, safe to drop) and press Enter.
--point: needs arm_board.yaml; each Enter prints where the gripper tip is on the board (graveyard spot, travel height).
--refit: recompute arm_board.yaml from its saved joint angles with the current config (no arm needed).
Type q and Enter to stop.
"""
import argparse
import itertools
import math
import os
import sys
import threading
import time

import chess
import numpy as np
import yaml
from lerobot.model.kinematics import RobotKinematics

# Run as a script from the repo root, so make the project folders importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from control import kinematics  # noqa: E402

DEFAULT_SQUARES = ["a1", "h1", "h8", "a8"]
SCREEN_REFRESH_S = 0.1


def open_limp_follower(arm_connection: dict):
    """Connect to the follower's motors only and switch their torque off. Never switches it back on.

    lerobot's own robot.connect() turns the torque on at the end, and the arm could jump to an old target.
    """
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    robot = SO101Follower(SO101FollowerConfig(port=arm_connection["port"], id=arm_connection["id"]))
    robot.bus.connect()
    robot.bus.disable_torque()
    if not robot.bus.is_calibrated:
        robot.bus.disconnect(disable_torque=True)
        raise SystemExit(f"the motors don't match lerobot's calibration '{arm_connection['id']}': run lerobot-calibrate")
    return robot


def read_joint_angles(robot) -> dict[str, float]:
    """All 6 joints right now, in lerobot degrees (gripper 0-100)."""
    return {joint: float(angle) for joint, angle in robot.bus.sync_read("Present_Position").items()}


def wait_for_enter() -> dict:
    """Start waiting for the user to press Enter in the background; the result dict gets the typed text."""
    typed = {}

    def wait():
        typed["text"] = input().strip().lower()
    threading.Thread(target=wait, daemon=True).start()
    return typed


def record_held_pose(robot, instruction: str, arm_settings: dict, arm_model_only: dict,
                     board_to_robot: np.ndarray | None = None) -> dict[str, float] | None:
    """Show the gripper tip live until Enter, then return the joint angles averaged while held still. None on q."""
    print(f"\n{instruction}\nPress Enter when it's in place (q + Enter to stop).")
    typed = wait_for_enter()
    while "text" not in typed:
        tip, tilt = kinematics.gripper_position_for(read_joint_angles(robot), arm_settings, arm_model_only)
        line = f"tip from base x {tip[0]:+.3f} y {tip[1]:+.3f} z {tip[2]:+.3f} m, tilt {tilt:5.1f} deg"
        if board_to_robot is not None:
            on_board = robot_point_to_board(tip, board_to_robot)
            line += f" | on board x {on_board[0]:+.3f} y {on_board[1]:+.3f} height {on_board[2]:+.3f} m"
        print("\r" + line, end="", flush=True)
        time.sleep(SCREEN_REFRESH_S)
    print()
    if typed["text"] == "q":
        return None

    # Average a short burst of readings so a tiny wobble of the hand doesn't matter.
    readings = []
    end_time = time.monotonic() + arm_settings["calibration_sample_s"]
    while time.monotonic() < end_time:
        readings.append(read_joint_angles(robot))
    return {joint: float(np.mean([reading[joint] for reading in readings])) for joint in readings[0]}


def robot_point_to_board(point_from_robot: np.ndarray, board_to_robot: np.ndarray) -> np.ndarray:
    """A point measured from the robot's base, as (x, y, height) in the board frame."""
    return (np.linalg.inv(board_to_robot) @ np.append(point_from_robot, 1.0))[:3]


def fit_board(board_points: np.ndarray, robot_points: np.ndarray) -> np.ndarray:
    """The rotation plus shift (no stretching) that best moves the board points onto the measured robot points."""
    board_middle, robot_middle = board_points.mean(axis=0), robot_points.mean(axis=0)
    spread = (board_points - board_middle).T @ (robot_points - robot_middle)
    left, _, right = np.linalg.svd(spread)
    # Flip the last axis if needed so the result is a true rotation, not a mirror image.
    mirror_fix = np.diag([1.0, 1.0, np.sign(np.linalg.det(right.T @ left.T))])
    rotation = right.T @ mirror_fix @ left.T
    board_to_robot = np.eye(4)
    board_to_robot[:3, :3] = rotation
    board_to_robot[:3, 3] = robot_middle - rotation @ board_middle
    return board_to_robot


def fit_and_report(samples: list[dict], arm_settings: dict, board_settings: dict, arm_model_only: dict) -> dict:
    """Fit the board from the recorded samples, print how good the fit is and suggested config lines; return arm_board.yaml."""
    touch_height = arm_settings["grip_height_frac"] * board_settings["piece_height_m"]["pawn"]
    names = [sample["square"] for sample in samples]
    board_points, robot_points, tilts = [], [], []
    for sample in samples:
        tip, tilt = kinematics.gripper_position_for(sample["joint_angles"], arm_settings, arm_model_only)
        sample["point_from_robot"] = [float(value) for value in tip]
        centre = kinematics.square_center_on_board(chess.parse_square(sample["square"]), board_settings)
        board_points.append([centre[0], centre[1], touch_height])
        robot_points.append(tip)
        tilts.append(tilt)
    board_points, robot_points = np.array(board_points), np.array(robot_points)
    board_to_robot = fit_board(board_points, robot_points)
    rotation = board_to_robot[:3, :3]

    print("\nLeftover error at each square after the fit (should be a few mm at most):")
    square_corrections = []
    for name, board_point, robot_point in zip(names, board_points, robot_points):
        fitted = rotation @ board_point + board_to_robot[:3, 3]
        error_in_board_frame = rotation.T @ (robot_point - fitted)
        size = np.linalg.norm(error_in_board_frame)
        warning = "  <-- larger than calibration_max_error_m" if size > arm_settings["calibration_max_error_m"] else ""
        print(f"  {name}: {size * 1000:5.1f} mm (along files {error_in_board_frame[0] * 1000:+.1f}, "
              f"along ranks {error_in_board_frame[1] * 1000:+.1f}, height {error_in_board_frame[2] * 1000:+.1f}){warning}")
        square_corrections.append({"board_x_y": [float(board_point[0]), float(board_point[1])],
                                   "correction_x_y": [float(error_in_board_frame[0]), float(error_in_board_frame[1])]})

    print("\nDistances between the squares you touched (measured vs real board):")
    for (first, first_board, first_robot), (second, second_board, second_robot) in itertools.combinations(
            zip(names, board_points, robot_points), 2):
        measured = np.linalg.norm(first_robot - second_robot)
        expected = np.linalg.norm(first_board - second_board)
        print(f"  {first}-{second}: {measured * 1000:6.1f} mm vs {expected * 1000:6.1f} mm "
              f"({(measured - expected) * 1000:+.1f})")
    board_tilt = math.degrees(math.acos(np.clip(rotation[2, 2], -1.0, 1.0)))
    print(f"\nThe board is tilted {board_tilt:.1f} deg from the arm's level.")

    print_suggested_tool_pose(names, robot_points, tilts, samples)
    return {"board_to_robot_transform": board_to_robot.tolist(), "touch_height_m": float(touch_height),
            "samples": samples, "square_corrections": square_corrections}


def print_suggested_tool_pose(names: list[str], robot_points: np.ndarray, tilts: list[float], samples: list[dict]) -> None:
    """Print the tilt and reach you held at the nearest and farthest ranks, as lines to copy into config/arm.yaml."""
    print("\nHow you held the gripper:")
    for name, point, tilt, sample in zip(names, robot_points, tilts, samples):
        print(f"  {name}: tilt {tilt:5.1f} deg, reach {math.hypot(point[0], point[1]):.3f} m, "
              f"wrist_roll {sample['joint_angles']['wrist_roll']:+.1f} deg")
    ranks = [chess.square_rank(chess.parse_square(name)) for name in names]
    near = [index for index, rank in enumerate(ranks) if rank == min(ranks)]
    far = [index for index, rank in enumerate(ranks) if rank == max(ranks)]
    if min(ranks) == max(ranks):
        return
    reach = [math.hypot(point[0], point[1]) for point in robot_points]
    print("\nSuggested lines for config/arm.yaml (copy by hand if they look right):")
    print(f"pitch_near_deg: {np.mean([tilts[i] for i in near]):.1f}")
    print(f"pitch_far_deg: {np.mean([tilts[i] for i in far]):.1f}")
    print(f"reach_near_m: {np.mean([reach[i] for i in near]):.3f}")
    print(f"reach_far_m: {np.mean([reach[i] for i in far]):.3f}")
    print(f"wrist_roll_deg: {np.mean([sample['joint_angles']['wrist_roll'] for sample in samples]):.1f}")


def save_yaml(contents: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("# Written by calibration/arm_board.py. Do not edit by hand; rerun the script.\n")
        yaml.safe_dump(contents, f, sort_keys=False)
    print(f"Saved {path}")


def main(command_line_args: list[str] | None = None) -> int:
    """Run the calibration described at the top of this file."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--squares", nargs="+", default=DEFAULT_SQUARES, help="squares to stand the pawn on")
    what = parser.add_mutually_exclusive_group()
    what.add_argument("--park", action="store_true", help="record the resting pose")
    what.add_argument("--point", action="store_true", help="measure spots on the board (writes nothing)")
    what.add_argument("--refit", action="store_true", help="refit from the saved joint angles (no arm)")
    parser.add_argument("--output-folder", default="calibration/data")
    args = parser.parse_args(command_line_args)
    for name in args.squares:
        if name not in chess.SQUARE_NAMES:
            parser.error(f"not a square: {name}")
    if not (args.park or args.point or args.refit) and len(args.squares) < 3:
        parser.error("the board fit needs at least 3 squares (not all on one line)")
    with open("config/arm.yaml") as f:
        arm_settings = yaml.safe_load(f)
    with open("config/board.yaml") as f:
        board_settings = yaml.safe_load(f)
    with open("config/hardware.yaml") as f:
        arm_connection = yaml.safe_load(f)["arm"]
    arm_model_only = {"arm_model": RobotKinematics(arm_settings["urdf_path"], arm_settings["target_frame"],
                                                   kinematics.BODY_JOINTS)}
    board_file = os.path.join(args.output_folder, "arm_board.yaml")

    if args.refit:
        if not os.path.exists(board_file):
            parser.error(f"{board_file} is missing: record the squares first")
        with open(board_file) as f:
            samples = yaml.safe_load(f)["samples"]
        save_yaml(fit_and_report(samples, arm_settings, board_settings, arm_model_only), board_file)
        return 0

    board_to_robot = None
    if args.point:
        if not os.path.exists(board_file):
            parser.error(f"{board_file} is missing: record the squares first")
        with open(board_file) as f:
            board_to_robot = np.array(yaml.safe_load(f)["board_to_robot_transform"])

    robot = open_limp_follower(arm_connection)
    print("The arm's motors are OFF: hold the arm, it is limp.")
    try:
        if args.park:
            pose = record_held_pose(robot, "Move the arm to its resting pose: out of the camera's view, "
                                    "and where it can safely drop.", arm_settings, arm_model_only)
            if pose is None:
                return 1
            save_yaml(pose, os.path.join(args.output_folder, "park_pose.yaml"))
        elif args.point:
            while True:
                pose = record_held_pose(robot, "Hold the gripper tip at the spot to measure.", arm_settings,
                                        arm_model_only, board_to_robot)
                if pose is None:
                    return 0
                tip, _ = kinematics.gripper_position_for(pose, arm_settings, arm_model_only)
                on_board = robot_point_to_board(tip, board_to_robot)
                print(f"On the board: x {on_board[0]:+.4f} m, y {on_board[1]:+.4f} m, "
                      f"height above the board {on_board[2]:+.4f} m")
        else:
            samples = []
            for name in args.squares:
                pose = record_held_pose(robot, f"Stand the pawn (held in the jaws) on the centre of {name}, "
                                        "gripper held the way you'd pick a piece.", arm_settings, arm_model_only)
                if pose is None:
                    return 1
                samples.append({"square": name, "joint_angles": pose})
            save_yaml(fit_and_report(samples, arm_settings, board_settings, arm_model_only), board_file)
    finally:
        robot.bus.disconnect(disable_torque=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
