"""Square -> base-frame pose -> joint angles, via lerobot RobotKinematics (placo) on the SO101 URDF."""
import numpy as np


def load_arm_calib(data_dir: str, arm_cfg: dict, allow_nominal: bool) -> dict:
    """Load {"kin", "T_board_base", "offsets", "park"}; nominal fallback only if allowed, else FileNotFoundError."""
    raise NotImplementedError


def square_xy(sq: int, board_cfg: dict) -> tuple[float, float]:
    """Square center in the board frame (m); raise ValueError if sq not in 0-63."""
    raise NotImplementedError


def board_to_base(xy: tuple[float, float], z: float, acal: dict) -> np.ndarray:
    """Board-frame (x, y) + height z -> (3,) base-frame point (m), with interpolated offsets."""
    raise NotImplementedError


def tool_pitch(p_base: np.ndarray, arm_cfg: dict) -> float:
    """Tool pitch (deg from horizontal, + down) for a base-frame point, from the reach schedule."""
    raise NotImplementedError


def ik(p_base: np.ndarray, pitch_deg: float, seed: dict[str, float], arm_cfg: dict, acal: dict) -> dict[str, float] | None:
    """TCP position + pitch (seeded solve) -> 5 body joints in lerobot deg; None if FK check fails or out of limits."""
    raise NotImplementedError


def fk(joints: dict[str, float], arm_cfg: dict, acal: dict) -> tuple[np.ndarray, float]:
    """5 body joints in lerobot deg -> (TCP (3,) base m, pitch deg)."""
    raise NotImplementedError
