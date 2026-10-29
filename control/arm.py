"""Drives the real SO101 arm through lerobot, plus a pretend arm for working without hardware."""


class Arm:
    """The real arm. Joint angles are in degrees; the gripper is 0-100."""

    def __init__(self, hardware_settings: dict, arm_settings: dict):
        raise NotImplementedError

    def connect(self) -> None:
        """Connect to the arm. Raises ConnectionError if it isn't plugged in."""
        raise NotImplementedError

    def read_joint_angles(self) -> dict[str, float]:
        """Return where all six joints are right now."""
        raise NotImplementedError

    def set_joint_targets(self, targets: dict[str, float]) -> dict[str, float]:
        """Tell some or all joints where to go. Values past a joint's safe limit are cut back to the limit.

        Returns what was actually sent. Doesn't wait for the arm to arrive.
        """
        raise NotImplementedError

    def set_gripper_strength(self, torque_limit: int) -> None:
        """Set how hard the gripper may squeeze (0-1000), so it stops on the piece instead of crushing it."""
        raise NotImplementedError

    def turn_off_motors(self) -> None:
        """Let the arm go limp. Never raises."""
        raise NotImplementedError

    def disconnect(self) -> None:
        """Close the connection. Safe to call more than once."""
        raise NotImplementedError


class MockArm:
    """A pretend arm with the same methods as Arm. It simply remembers the last targets it was given."""

    def __init__(self, hardware_settings: dict, arm_settings: dict, starting_joint_angles: dict[str, float]):
        raise NotImplementedError

    def connect(self) -> None:
        """Does nothing."""
        raise NotImplementedError

    def read_joint_angles(self) -> dict[str, float]:
        """Return the last targets it was given."""
        raise NotImplementedError

    def set_joint_targets(self, targets: dict[str, float]) -> dict[str, float]:
        """Cut values back to the safe limits, remember them, and return them."""
        raise NotImplementedError

    def set_gripper_strength(self, torque_limit: int) -> None:
        """Remember the value."""
        raise NotImplementedError

    def turn_off_motors(self) -> None:
        """Remember that the motors were turned off."""
        raise NotImplementedError

    def disconnect(self) -> None:
        """Does nothing."""
        raise NotImplementedError
