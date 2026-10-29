"""find_human_move (occupancy before/after -> human's legal move) and get_robot_move (move -> list of move/remove ops: captures, castling, en passant)."""
import chess
import numpy as np


def expected_board_state(board: chess.Board) -> np.ndarray:
    """(8,8) bool [rank, file], True where the board has a piece."""
    occ = np.zeros(64, dtype=bool)
    occ[list(board.piece_map())] = True
    return occ.reshape(8, 8)


def find_human_move(board: chess.Board, before: np.ndarray, after: np.ndarray) -> list[chess.Move]:
    """Legal moves whose changed squares equal before ^ after (promotions as queen); >1 means ambiguous."""
    if np.shape(before) != (8, 8) or np.shape(after) != (8, 8):
        return []
    seen = np.asarray(before, dtype=bool) ^ np.asarray(after, dtype=bool)
    now = expected_board_state(board)
    found = []
    for move in board.legal_moves:
        if move.promotion not in (None, chess.QUEEN):  # vision can't tell the piece; assume a queen
            continue
        nxt = board.copy(stack=False)
        nxt.push(move)
        if np.array_equal(now ^ expected_board_state(nxt), seen):
            found.append(move)
    return found


def get_robot_move(board: chess.Board, move: chess.Move) -> list[tuple]:
    """Move -> [("remove", sq) ..., ("move", from_sq, to_sq) ...]; raise ValueError if illegal."""
    if move not in board.legal_moves:
        raise ValueError(f"illegal move {move.uci()} in {board.fen()}")
    frm, to = move.from_square, move.to_square
    if board.is_castling(move):
        # king lands on g/c, rook on f/d; computed, because python-chess also accepts king-takes-rook (e1h1)
        rank = chess.square_rank(frm)
        k_file, r_from, r_to = (6, 7, 5) if board.is_kingside_castling(move) else (2, 0, 3)
        return [("move", frm, chess.square(k_file, rank)),
                ("move", chess.square(r_from, rank), chess.square(r_to, rank))]
    if board.is_en_passant(move):
        # the captured pawn sits beside the start square, not on the target square
        return [("remove", chess.square(chess.square_file(to), chess.square_rank(frm))), ("move", frm, to)]
    if board.is_capture(move):
        return [("remove", to), ("move", frm, to)]
    return [("move", frm, to)]
