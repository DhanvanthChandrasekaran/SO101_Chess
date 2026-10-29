"""Talks to the Stockfish chess program to pick the robot's moves."""
import shutil

import chess
import chess.engine


class GameOverError(ValueError):
    """Raised when someone asks for a move after the game has already ended."""


def open_engine(engine_cfg: dict) -> chess.engine.SimpleEngine:
    """Start Stockfish and set how strongly it plays. Raises FileNotFoundError if Stockfish isn't installed."""
    stockfish_path = shutil.which(engine_cfg["path"])
    if stockfish_path is None:
        raise FileNotFoundError(f"Can't find Stockfish at {engine_cfg['path']}")
    stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        stockfish.configure({"Skill Level": engine_cfg["skill_level"]})
    except Exception:
        # Setting the strength failed: close Stockfish so it isn't left running, then report the error.
        stockfish.quit()
        raise
    return stockfish


def best_move(stockfish: chess.engine.SimpleEngine, board: chess.Board, engine_cfg: dict) -> chess.Move:
    """Ask Stockfish for the best move in this position. The board itself is not changed.

    Raises GameOverError if the game has already ended.
    """
    if board.is_game_over():
        raise GameOverError(f"The game is over: {board.result()}")
    thinking_time = chess.engine.Limit(time=engine_cfg["think_time_s"])
    return stockfish.play(board, thinking_time).move


def close_engine(stockfish: chess.engine.SimpleEngine) -> None:
    """Shut Stockfish down. Safe to call more than once."""
    try:
        stockfish.quit()
    except chess.engine.EngineTerminatedError:
        pass  # it was already closed
