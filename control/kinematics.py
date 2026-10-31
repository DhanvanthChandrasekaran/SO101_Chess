"""Converts "put the gripper here" into arm joint angles and back, using lerobot's arm model of the SO101."""
import math
import os

import numpy as np
import yaml
from lerobot.model.kinematics import RobotKinematics

BODY_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
PAN_LINK = "shoulder_link"  # URDF link whose origin lies on the vertical pan axis
# Rough board placement for --mock (docs/INTERFACES.md §2.7): board +x = base -y, board +y = base +x.
BOARD_FACING_ROBOT = np.array([[0.0, 1.0, 0.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0]])


def load_arm_calibration(calibration_folder: str, arm_settings: dict, allow_rough_defaults: bool) -> dict:
    """Load the arm model and where the board sits relative to the robot.

    If the calibration files are missing, rough defaults from the settings are used only when
    `allow_rough_defaults` is True (the --mock mode); otherwise raises FileNotFoundError.
    """
    urdf_path = arm_settings["urdf_path"]
    if not urdf_path or not os.path.exists(urdf_path):
        raise FileNotFoundError(f"arm model (URDF) not found: {urdf_path}")
    arm_model = RobotKinematics(urdf_path, arm_settings["target_frame"], BODY_JOINTS)
    rough_defaults = arm_settings.get("nominal") or {}

    board_file = os.path.join(calibration_folder, "arm_board.yaml")
    if os.path.exists(board_file):
        with open(board_file) as f:
            arm_board = yaml.safe_load(f)
        board_to_robot_transform = np.array(arm_board["board_to_robot_transform"], dtype=np.float64)
        square_corrections = [(tuple(entry["board_x_y"]), tuple(entry["correction_x_y"]))
                              for entry in arm_board["square_corrections"]]
    else:
        board_origin = rough_defaults.get("board_origin_base_m")
        if not allow_rough_defaults or board_origin is None or None in board_origin:
            raise FileNotFoundError(f"{board_file} is missing: run calibration/arm_board.py")
        board_to_robot_transform = np.eye(4)
        board_to_robot_transform[:3, :3] = BOARD_FACING_ROBOT
        board_to_robot_transform[:3, 3] = board_origin
        square_corrections = []

    park_file = os.path.join(calibration_folder, "park_pose.yaml")
    if os.path.exists(park_file):
        with open(park_file) as f:
            park_pose = yaml.safe_load(f)
    else:
        park_pose = rough_defaults.get("park_pose_deg")
        if not allow_rough_defaults or park_pose is None:
            raise FileNotFoundError(f"{park_file} is missing: run calibration/arm_board.py --park")
    park_pose = {joint: float(angle) for joint, angle in park_pose.items()}

    return {"arm_model": arm_model, "board_to_robot_transform": board_to_robot_transform,
            "square_corrections": square_corrections, "park_pose": park_pose}


def square_center_on_board(square: int, board_settings: dict) -> tuple[float, float]:
    """Return the (x, y) centre of a square in metres, measured from the outer corner of a1.

    Raises ValueError if the square number isn't 0-63.
    """
    if not 0 <= square <= 63:
        raise ValueError(f"square must be 0-63, got {square}")
    file_index, rank_index = square % 8, square // 8
    size = board_settings["square_size_m"]
    return ((file_index + 0.5) * size, (rank_index + 0.5) * size)


def board_point_to_robot(board_x_y: tuple[float, float], height_above_board: float,
                         arm_calibration: dict) -> np.ndarray:
    """Convert a spot on the board (metres) plus a height into a point measured from the robot's base."""
    correction_x, correction_y = correction_at(board_x_y, arm_calibration["square_corrections"])
    board_point = np.array([board_x_y[0] + correction_x, board_x_y[1] + correction_y, height_above_board, 1.0])
    return (arm_calibration["board_to_robot_transform"] @ board_point)[:3]


def correction_at(board_x_y: tuple[float, float], square_corrections: list) -> tuple[float, float]:
    """Blend the measured corrections, trusting nearby ones more (weight 1 / distance squared). (0, 0) if none."""
    if not square_corrections:
        return (0.0, 0.0)
    weights, corrections = [], []
    for measured_x_y, correction in square_corrections:
        distance_squared = (board_x_y[0] - measured_x_y[0]) ** 2 + (board_x_y[1] - measured_x_y[1]) ** 2
        if distance_squared < 1e-12:  # exactly on a measured square: use its own correction
            return tuple(correction)
        weights.append(1.0 / distance_squared)
        corrections.append(correction)
    blended = np.average(np.array(corrections), axis=0, weights=weights)
    return (float(blended[0]), float(blended[1]))


def gripper_tilt_for(point_from_robot: np.ndarray, arm_settings: dict) -> float:
    """How far the gripper should tilt down (degrees from level) to reach this point: steeper close, flatter far."""
    schedule = [arm_settings[key] for key in ("reach_near_m", "reach_far_m", "pitch_near_deg", "pitch_far_deg")]
    if None in schedule:
        raise ValueError("arm.yaml: reach_near_m, reach_far_m, pitch_near_deg and pitch_far_deg must be set")
    reach_near, reach_far, pitch_near, pitch_far = schedule
    reach = math.hypot(point_from_robot[0], point_from_robot[1])
    # np.interp holds the end values outside the near..far range.
    return float(np.interp(reach, [reach_near, reach_far], [pitch_near, pitch_far]))


def joint_angles_for(point_from_robot: np.ndarray, gripper_tilt: float, starting_guess: dict[str, float],
                     arm_settings: dict, arm_calibration: dict) -> dict[str, float] | None:
    """Find joint angles (degrees) that put the gripper tip at this point with this tilt.

    `starting_guess` is where the search starts, usually the arm's current angles.
    Returns None if the arm can't reach it or a joint would go past its limit.
    """
    arm_model = arm_calibration["arm_model"]
    point = np.asarray(point_from_robot, dtype=np.float64)
    tool_rotation = tool_rotation_for(point, gripper_tilt, arm_settings["wrist_roll_deg"],
                                      arm_settings["tcp_offset_m"], arm_model)
    if tool_rotation is None:
        return None
    # IK steers the URDF's gripper frame, which sits tcp_offset_m away from the tip we care about.
    target_pose = np.eye(4)
    target_pose[:3, :3] = tool_rotation
    target_pose[:3, 3] = point - tool_rotation @ np.array(arm_settings["tcp_offset_m"])

    guess = np.array([starting_guess[joint] for joint in BODY_JOINTS], dtype=np.float64)
    solution = arm_model.inverse_kinematics(guess, target_pose, position_weight=1.0,
                                            orientation_weight=arm_settings["ik_orientation_weight"],
                                            max_iters=arm_settings["ik_max_iters"])
    joint_angles = {joint: float(angle) for joint, angle in zip(BODY_JOINTS, solution)}

    # The solver always returns something; only accept it if running it forward lands on the target.
    reached_point, reached_tilt = gripper_position_for(joint_angles, arm_settings, arm_calibration)
    if np.linalg.norm(reached_point - point) > arm_settings["ik_pos_tol_m"]:
        return None
    if abs(reached_tilt - gripper_tilt) > arm_settings["ik_pitch_tol_deg"]:
        return None
    for joint, (lowest, highest) in arm_settings["joint_limits_deg"].items():
        if joint not in joint_angles:
            continue
        if (lowest is not None and joint_angles[joint] < lowest) or (highest is not None and joint_angles[joint] > highest):
            return None
    return joint_angles


def tool_rotation_for(point: np.ndarray, gripper_tilt: float, wrist_roll: float, tcp_offset: list[float],
                      arm_model: RobotKinematics) -> np.ndarray | None:
    """The gripper orientation that faces the point from the pan axis, tilted down by `gripper_tilt`, with this wrist roll.

    None if the point is so close to the pan axis that no direction of the arm reaches it.
    """
    if wrist_roll is None:
        raise ValueError("arm.yaml: wrist_roll_deg must be set")
    # With every joint at 0 except the wrist roll, the jaws point straight out along +x and level.
    level_pose = arm_model.forward_kinematics(np.array([0.0, 0.0, 0.0, 0.0, wrist_roll]))
    pan_axis = arm_model.robot.get_T_world_frame(PAN_LINK)[:3, 3]
    # The arm swings in a vertical plane through the pan axis, but a rolled wrist can put the tip a little
    # to one side of that plane. Turn the plane so that the tip, not the plane, lands on the point.
    tip_when_level = level_pose[:3, 3] + level_pose[:3, :3] @ np.array(tcp_offset)
    sideways_offset = tip_when_level[1] - pan_axis[1]
    distance_from_pan_axis = math.hypot(point[0] - pan_axis[0], point[1] - pan_axis[1])
    if distance_from_pan_axis <= abs(sideways_offset):
        return None  # no direction of the arm puts the tip here
    yaw = (math.atan2(point[1] - pan_axis[1], point[0] - pan_axis[0])
           - math.asin(sideways_offset / distance_from_pan_axis))
    tilt = math.radians(gripper_tilt)
    # Turning about +y by a positive angle points +x downward, so this tilts the jaws down.
    tilt_down = np.array([[math.cos(tilt), 0.0, math.sin(tilt)],
                          [0.0, 1.0, 0.0],
                          [-math.sin(tilt), 0.0, math.cos(tilt)]])
    turn_to_yaw = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                            [math.sin(yaw), math.cos(yaw), 0.0],
                            [0.0, 0.0, 1.0]])
    return turn_to_yaw @ tilt_down @ level_pose[:3, :3]


def gripper_position_for(joint_angles: dict[str, float], arm_settings: dict,
                         arm_calibration: dict) -> tuple[np.ndarray, float]:
    """The reverse: where the gripper tip ends up (metres from the base) and its tilt, for these joint angles."""
    angles = np.array([joint_angles[joint] for joint in BODY_JOINTS], dtype=np.float64)
    frame_pose = arm_calibration["arm_model"].forward_kinematics(angles)
    rotation = frame_pose[:3, :3]
    tip = frame_pose[:3, 3] + rotation @ np.array(arm_settings["tcp_offset_m"])
    pointing_direction = rotation[:, 2]  # the gripper frame's +z points out of the jaws
    tilt = math.degrees(math.asin(float(np.clip(-pointing_direction[2], -1.0, 1.0))))
    return tip, tilt
