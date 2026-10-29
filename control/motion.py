"""Plans and runs the arm's path for each step of a move: pick up, carry, put down, and park out of the way."""
from control import kinematics  # noqa: F401  (used by plan_waypoints)


def plan_waypoints(steps: list[tuple], piece_heights: dict[int, float], settings: dict) -> list[tuple]:
    """Turn the steps from get_robot_move, like ("move", e2, e4), into the arm poses and gripper open/close needed to do them.

    Everything is planned before the arm moves. Raises ValueError if any pose is off the board area,
    too low, or out of reach.
    """
    raise NotImplementedError


def run_waypoints(arm, waypoints: list[tuple], settings: dict) -> None:
    """Move the arm through the waypoints slowly and smoothly. Raises RuntimeError if the arm doesn't get there."""
    raise NotImplementedError


def park(arm, settings: dict) -> None:
    """Move the arm to its recorded resting pose, out of the camera's view."""
    raise NotImplementedError
