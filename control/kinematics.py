"""Converts "put the gripper here" into arm joint angles and back, using lerobot's arm model of the SO101."""
import numpy as np


def load_arm_calibration(calibration_folder: str, arm_settings: dict, allow_rough_defaults: bool) -> dict:
    """Load the arm model and where the board sits relative to the robot.

    If the calibration files are missing, rough defaults from the settings are used only when
    `allow_rough_defaults` is True (the --mock mode); otherwise raises FileNotFoundError.
    """
    raise NotImplementedError


def square_center_on_board(square: int, board_settings: dict) -> tuple[float, float]:
    """Return the (x, y) centre of a square in metres, measured from the outer corner of a1.

    Raises ValueError if the square number isn't 0-63.
    """
    raise NotImplementedError


def board_point_to_robot(board_x_y: tuple[float, float], height_above_board: float,
                         arm_calibration: dict) -> np.ndarray:
    """Convert a spot on the board (metres) plus a height into a point measured from the robot's base."""
    raise NotImplementedError


def gripper_tilt_for(point_from_robot: np.ndarray, arm_settings: dict) -> float:
    """How far the gripper should tilt down (degrees from level) to reach this point: steeper close, flatter far."""
    raise NotImplementedError


def joint_angles_for(point_from_robot: np.ndarray, gripper_tilt: float, starting_guess: dict[str, float],
                     arm_settings: dict, arm_calibration: dict) -> dict[str, float] | None:
    """Find joint angles (degrees) that put the gripper tip at this point with this tilt.

    `starting_guess` is where the search starts, usually the arm's current angles.
    Returns None if the arm can't reach it or a joint would go past its limit.
    """
    raise NotImplementedError


def gripper_position_for(joint_angles: dict[str, float], arm_settings: dict,
                         arm_calibration: dict) -> tuple[np.ndarray, float]:
    """The reverse: where the gripper tip ends up (metres from the base) and its tilt, for these joint angles."""
    raise NotImplementedError
