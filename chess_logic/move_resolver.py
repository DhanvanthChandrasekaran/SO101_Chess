"""match_move (occupancy before/after -> human's legal move) and ops_for (move -> list of move/remove ops: captures, castling, en passant)."""
import chess
import numpy as np


def expected_occupancy(board: chess.Board) -> np.ndarray:
    """(8,8) bool [rank, file], True where the board has a piece."""
    raise NotImplementedError


def match_move(board: chess.Board, before: np.ndarray, after: np.ndarray,
               change: np.ndarray | None = None, margin: float = 0.0) -> list[chess.Move]:
    """Legal moves whose changed squares equal before ^ after; ties broken by change at to_square if clear by margin."""
    raise NotImplementedError


def ops_for(board: chess.Board, move: chess.Move) -> list[tuple]:
    """Move -> [("remove", sq) ..., ("move", from_sq, to_sq) ...]; raise ValueError if illegal."""
    raise NotImplementedError
