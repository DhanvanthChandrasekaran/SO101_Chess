"""pytest: move matching, move ops, castling, en passant."""
import shutil

import chess
import pytest
import yaml

from chess_logic import engine

with open("config/game.yaml") as f:
    ENGINE_CFG = {**yaml.safe_load(f)["engine"], "think_time_s": 0.1}

needs_stockfish = pytest.mark.skipif(shutil.which(ENGINE_CFG["path"]) is None, reason="Stockfish binary missing")


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
