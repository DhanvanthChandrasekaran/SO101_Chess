"""Waypoints/trajectory: pick, place, remove, park."""
from control import kinematics  # noqa: F401  (used by plan_ops)


def plan_ops(ops: list[tuple], heights: dict[int, float], ctx: dict) -> list[tuple]:
    """Ops -> steps [("joints", {...}) | ("gripper", "open"|"close")]; raise ValueError if any target is unsafe/unreachable."""
    raise NotImplementedError


def execute(arm, steps: list[tuple], ctx: dict) -> None:
    """Run steps with speed-limited joint interpolation; raise RuntimeError if a step doesn't settle."""
    raise NotImplementedError


def park(arm, ctx: dict) -> None:
    """Move to the recorded park pose via travel height."""
    raise NotImplementedError
