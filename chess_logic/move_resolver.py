"""Works out the human's move from what the camera saw, and turns the robot's move into steps for the arm."""
import chess
import numpy as np


def expected_board_state(board: chess.Board) -> np.ndarray:
    """Return an 8x8 grid of True/False: True where the game says a piece should be standing.

    Row 0 is rank 1 (the robot's side) and column 0 is the a-file.
    """
    has_piece = np.zeros(64, dtype=bool)
    squares_with_pieces = list(board.piece_map())
    has_piece[squares_with_pieces] = True
    # Squares are numbered a1=0, b1=1 ... h8=63, so folding into rows of 8 puts rank 1 in row 0.
    return has_piece.reshape(8, 8)


def find_human_move(board: chess.Board, before: np.ndarray, after: np.ndarray) -> list[chess.Move]:
    """Work out the human's move by comparing the camera's grid from before and after their turn.

    Returns every move that fits: one move means found, several means the human must say which one,
    and an empty list means nothing fits (camera mistake or a knocked piece).
    """
    if np.shape(before) != (8, 8) or np.shape(after) != (8, 8):
        return []

    squares_seen_changing = np.asarray(before, dtype=bool) != np.asarray(after, dtype=bool)
    pieces_now = expected_board_state(board)

    possible_moves = []
    for move in board.legal_moves:
        # The camera can't tell which piece a pawn turned into, so we assume a queen.
        if move.promotion is not None and move.promotion != chess.QUEEN:
            continue
        # Try the move on a copy, so the real game is not changed.
        board_after_move = board.copy(stack=False)
        board_after_move.push(move)
        squares_this_move_changes = pieces_now != expected_board_state(board_after_move)
        if np.array_equal(squares_this_move_changes, squares_seen_changing):
            possible_moves.append(move)
    return possible_moves


def get_robot_move(board: chess.Board, move: chess.Move) -> list[tuple]:
    """Turn the robot's chess move into a list of steps for the arm, in the order to do them.

    A step is ("remove", square) to take a captured piece off the board,
    or ("move", start_square, end_square) to pick a piece up and put it down.
    `board` is the game just before the robot's move. Raises ValueError if the move is not allowed.
    """
    if move not in board.legal_moves:
        raise ValueError(f"The move {move.uci()} is not allowed in this position: {board.fen()}")

    start = move.from_square
    end = move.to_square
    row = chess.square_rank(start)  # 0 for white's back row, 7 for black's

    if board.is_castling(move):
        # The king and a rook both move. We work out where they land ourselves instead of using `end`,
        # because python-chess also accepts castling written as "king onto its own rook" (e.g. e1h1).
        if board.is_kingside_castling(move):
            king_column, rook_start_column, rook_end_column = 6, 7, 5  # king to g, rook from h to f
        else:
            king_column, rook_start_column, rook_end_column = 2, 0, 3  # king to c, rook from a to d
        return [("move", start, chess.square(king_column, row)),
                ("move", chess.square(rook_start_column, row), chess.square(rook_end_column, row))]

    # En passant must be checked before normal captures, because python-chess counts it as a capture too.
    if board.is_en_passant(move):
        # Special pawn capture: the captured pawn is not on the end square but right beside the start square.
        captured_pawn = chess.square(chess.square_file(end), row)
        return [("remove", captured_pawn), ("move", start, end)]

    if board.is_capture(move):
        # Clear the end square first, so there is room to put our piece down.
        return [("remove", end), ("move", start, end)]

    # A normal move. A pawn becoming a queen also lands here; the human swaps in the queen.
    return [("move", start, end)]
