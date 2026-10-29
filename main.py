"""Runs a chess game: watch the human's move, ask Stockfish for a reply, move the arm, check the board."""


def main(command_line_args: list[str] | None = None) -> int:
    """Play one game. --mock uses the pretend arm; --images DIR reads saved photos instead of the camera.

    Returns 0 when the game ends normally and 1 if it was stopped.
    """
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
