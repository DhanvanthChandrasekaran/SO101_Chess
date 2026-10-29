"""Stockfish wrapper: board -> best move."""
import chess
import chess.engine


def open_engine(engine_cfg: dict) -> chess.engine.SimpleEngine:
    """Start Stockfish at engine_cfg["path"] with the configured skill level."""
    raise NotImplementedError


def best_move(eng: chess.engine.SimpleEngine, board: chess.Board, engine_cfg: dict) -> chess.Move:
    """Best move for the side to move; raise ValueError if the game is over."""
    raise NotImplementedError


def close_engine(eng: chess.engine.SimpleEngine) -> None:
    """Quit the engine; safe to call twice."""
    raise NotImplementedError
