"""lerobot SO101 wrapper + MockArm for no-hardware development."""


class Arm:
    """Real SO101 follower via lerobot. Joint dicts use bare names, body joints in deg, gripper 0-100."""

    def __init__(self, hw_cfg: dict, arm_cfg: dict):
        raise NotImplementedError

    def connect(self) -> None:
        """Connect to the follower; raise ConnectionError if the port is missing."""
        raise NotImplementedError

    def read(self) -> dict[str, float]:
        """Current position of all 6 joints."""
        raise NotImplementedError

    def send(self, target: dict[str, float]) -> dict[str, float]:
        """Send a subset of joints, clamped to limits; return what was sent."""
        raise NotImplementedError

    def set_gripper_torque(self, limit: int) -> None:
        """Write gripper Max_Torque_Limit (0-1000)."""
        raise NotImplementedError

    def torque_off(self) -> None:
        """Disable torque on all motors; never raises."""
        raise NotImplementedError

    def disconnect(self) -> None:
        """Disconnect; safe to call twice."""
        raise NotImplementedError


class MockArm:
    """Same interface as Arm; read() returns the last sent (clamped) target."""

    def __init__(self, hw_cfg: dict, arm_cfg: dict, start: dict[str, float]):
        raise NotImplementedError

    def connect(self) -> None:
        """No-op."""
        raise NotImplementedError

    def read(self) -> dict[str, float]:
        """Last commanded position of all 6 joints."""
        raise NotImplementedError

    def send(self, target: dict[str, float]) -> dict[str, float]:
        """Clamp and store the target; return what was stored."""
        raise NotImplementedError

    def set_gripper_torque(self, limit: int) -> None:
        """Record the limit."""
        raise NotImplementedError

    def torque_off(self) -> None:
        """Record torque off."""
        raise NotImplementedError

    def disconnect(self) -> None:
        """No-op."""
        raise NotImplementedError
