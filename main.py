"""Entry point: game loop + verify/retry/ask-human. Only file that wires modules together."""


def main(argv: list[str] | None = None) -> int:
    """Run a game (--mock: MockArm, --images DIR: offline frames); return 0 on normal end, 1 on abort."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
