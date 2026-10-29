"""pytest: move matching, move ops, castling, en passant."""
import shutil

import chess
import numpy as np
import pytest
import yaml

from chess_logic import engine
from chess_logic.move_resolver import expected_board_state, find_human_move, get_robot_move

with open("config/game.yaml") as f:
    ENGINE_CFG = {**yaml.safe_load(f)["engine"], "think_time_s": 0.1}

needs_stockfish = pytest.mark.skipif(shutil.which(ENGINE_CFG["path"]) is None, reason="Stockfish binary missing")

CASTLE_FEN = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R {} KQkq - 0 1"


def board_after(*ucis, fen=chess.STARTING_FEN):
    board = chess.Board(fen)
    for u in ucis:
        board.push_uci(u)
    return board


def grids(board, uci):
    """Perfect camera: occupancy before and after playing uci on board."""
    return expected_board_state(board), expected_board_state(board_after(uci, fen=board.fen()))


def found(board, uci):
    return sorted(m.uci() for m in find_human_move(board, *grids(board, uci)))


# --- expected_board_state ---

def test_expected_board_state_start():
    occ = expected_board_state(chess.Board())
    assert occ.shape == (8, 8) and occ.dtype == bool
    assert occ[[0, 1, 6, 7]].all() and not occ[2:6].any()


def test_expected_board_state_layout():
    occ = expected_board_state(board_after("e2e4"))
    assert not occ[1, 4] and occ[3, 4]  # [rank, file]: e2 empty, e4 full
    assert occ.reshape(64)[chess.E4]


# --- find_human_move ---

@pytest.mark.parametrize("board, uci", [
    (board_after("e2e4"), "g8f6"),                                       # quiet
    (board_after("e2e4", "d7d5", "b1c3"), "d5e4"),                       # capture: only d5 changes
    (chess.Board(CASTLE_FEN.format("b")), "e8g8"),                       # O-O
    (chess.Board(CASTLE_FEN.format("b")), "e8c8"),                       # O-O-O
    (board_after("e2e4", fen="4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1"), "d4e3"),  # en passant
    (chess.Board("4k3/8/8/8/8/8/p7/4K3 b - - 0 1"), "a2a1q"),            # promotion
    (chess.Board("4k3/8/8/8/8/8/1p6/R3K3 b - - 0 1"), "b2a1q"),          # capture-promotion
])
def test_find_human_move_unique(board, uci):
    assert found(board, uci) == [uci]


def test_find_human_move_promotion_is_queen():
    board = chess.Board("4k3/8/8/8/8/8/p7/4K3 b - - 0 1")
    assert found(board, "a2a1n") == ["a2a1q"]  # camera can't tell a knight from a queen


def test_find_human_move_ambiguous_capture():
    board = chess.Board("4k3/3p4/8/8/6p1/8/8/3QK3 w - - 0 1")  # queen d1 can take d7 or g4
    assert found(board, "d1g4") == ["d1d7", "d1g4"]


def test_find_human_move_noise():
    board = board_after("e2e4")
    before, after = grids(board, "g8f6")
    after[2, 0] = True  # phantom piece on a3
    assert find_human_move(board, before, after) == []


def test_find_human_move_nothing_changed():
    occ = expected_board_state(board_after("e2e4"))
    assert find_human_move(board_after("e2e4"), occ, occ.copy()) == []


def test_find_human_move_steady_misread():
    board = board_after("e2e4")
    before, after = grids(board, "g8f6")
    before[3, 0] = after[3, 0] = True  # a4 always misread as full
    assert [m.uci() for m in find_human_move(board, before, after)] == ["g8f6"]


def test_find_human_move_bad_input():
    board = chess.Board()
    assert find_human_move(board, np.zeros((8, 7), bool), np.zeros((8, 8), bool)) == []
    assert find_human_move(board, None, None) == []


def test_find_human_move_leaves_board():
    board = board_after("e2e4")
    fen = board.fen()
    found(board, "g8f6")
    assert board.fen() == fen


# --- get_robot_move ---

@pytest.mark.parametrize("board, uci, ops", [
    (chess.Board(), "e2e4", [("move", 12, 28)]),
    (board_after("e2e4", "d7d5"), "e4d5", [("remove", 35), ("move", 28, 35)]),
    (chess.Board(CASTLE_FEN.format("w")), "e1g1", [("move", 4, 6), ("move", 7, 5)]),
    (chess.Board(CASTLE_FEN.format("w")), "e1h1", [("move", 4, 6), ("move", 7, 5)]),   # king-takes-rook spelling
    (chess.Board(CASTLE_FEN.format("w")), "e1c1", [("move", 4, 2), ("move", 0, 3)]),
    (chess.Board(CASTLE_FEN.format("b")), "e8g8", [("move", 60, 62), ("move", 63, 61)]),
    (board_after("d7d5", fen="4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 1"), "e5d6", [("remove", 35), ("move", 36, 43)]),
    (chess.Board("k7/4P3/8/8/8/8/8/4K3 w - - 0 1"), "e7e8q", [("move", 52, 60)]),
])
def test_get_robot_move(board, uci, ops):
    fen = board.fen()
    assert get_robot_move(board, chess.Move.from_uci(uci)) == ops
    assert board.fen() == fen


def test_get_robot_move_illegal():
    with pytest.raises(ValueError):
        get_robot_move(chess.Board(), chess.Move.from_uci("e2e5"))


# --- engine ---

@pytest.fixture
def eng():
    e = engine.open_engine(ENGINE_CFG)
    yield e
    engine.close_engine(e)


@needs_stockfish
def test_best_move_start_position(eng):
    board = chess.Board()
    move = engine.best_move(eng, board, ENGINE_CFG)
    assert move in board.legal_moves
    assert board == chess.Board()


@needs_stockfish
def test_best_move_game_over_raises(eng):
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")  # fool's mate
    with pytest.raises(engine.GameOverError):
        engine.best_move(eng, board, ENGINE_CFG)


@needs_stockfish
def test_close_engine_twice():
    e = engine.open_engine(ENGINE_CFG)
    engine.close_engine(e)
    engine.close_engine(e)


def test_open_engine_bad_path():
    with pytest.raises(FileNotFoundError):
        engine.open_engine({**ENGINE_CFG, "path": "/nonexistent/stockfish"})
