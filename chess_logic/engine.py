"""Stockfish wrapper: board -> best move."""
import shutil

import chess
import chess.engine


class GameOverError(ValueError):
    """best_move called on a finished game."""


def open_engine(engine_cfg: dict) -> chess.engine.SimpleEngine:
    """Start Stockfish at engine_cfg["path"] with the configured skill level."""
    path = shutil.which(engine_cfg["path"])
    if path is None:
        raise FileNotFoundError(f"Stockfish not found: {engine_cfg['path']}")
    eng = chess.engine.SimpleEngine.popen_uci(path)
    try:
        eng.configure({"Skill Level": engine_cfg["skill_level"]})
    except Exception:
        eng.quit()
        raise
    return eng


def best_move(eng: chess.engine.SimpleEngine, board: chess.Board, engine_cfg: dict) -> chess.Move:
    """Best move for the side to move; raise GameOverError if the game is over."""
    if board.is_game_over():
        raise GameOverError(f"game is over: {board.result()}")
    return eng.play(board, chess.engine.Limit(time=engine_cfg["think_time_s"])).move


def close_engine(eng: chess.engine.SimpleEngine) -> None:
    """Quit the engine; safe to call twice."""
    try:
        eng.quit()
    except chess.engine.EngineTerminatedError:
        pass
