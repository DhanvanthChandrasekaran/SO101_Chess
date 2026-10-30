"""Live test of move recognition: you play both colours, press space after each move, and it names the move.

Usage: python tools/watch_moves.py FOLDER   (needs the camera and a threshold in config/vision.yaml; arm out of view)
Set up the starting position and press space. Then make a move, keep your hands out of view, press space.
Every recognised move and its photo is saved in FOLDER, so `collect_states.py check FOLDER` can score it later.
"""
import argparse
import os
import sys

import chess
import cv2
import matplotlib.pyplot as plt
import yaml

# Run as a script from the repo root, so make the project folders importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chess_logic.move_resolver import expected_board_state, find_human_move  # noqa: E402
from tools.collect_states import draw_position, load_calibration, mark_wrong_squares  # noqa: E402
from tools.vision_live import draw_board_grid, look_at_board  # noqa: E402
from vision import camera  # noqa: E402


def square_names(grid) -> str:
    """Names of the squares that are True in an 8x8 grid (row 0 = rank 1), e.g. 'e2 e4'."""
    return " ".join(chess.square_name(square) for square in chess.SQUARES if grid.reshape(64)[square]) or "none"


def describe_played_move(board: chess.Board, move: chess.Move) -> str:
    """A move that was just made, in words, e.g. 'white pawn e4 -> d5, taking the black pawn on d5'."""
    colour, other_colour = ("white", "black") if board.turn == chess.WHITE else ("black", "white")
    start, end = chess.square_name(move.from_square), chess.square_name(move.to_square)
    if board.is_castling(move):
        return f"{colour} castles ({'king' if chess.square_file(move.to_square) == 6 else 'queen'}side)"
    text = f"{colour} {chess.piece_name(board.piece_type_at(move.from_square))} {start} -> {end}"
    if board.is_en_passant(move):
        text += f", taking the {other_colour} pawn en passant"
    elif board.is_capture(move):
        text += f", taking the {other_colour} {chess.piece_name(board.piece_type_at(move.to_square))} on {end}"
    if move.promotion:
        text += ", promoting to a queen"
    return text


def ask_which_move(board: chess.Board, candidates: list[chess.Move]) -> chess.Move:
    """Several moves fit what the camera saw: list them and let the person type which one they made."""
    print("I couldn't tell which move you made. It could be:")
    for number, move in enumerate(candidates, start=1):
        print(f"  {number}. {describe_played_move(board, move)}")
    while True:
        answer = input("Type the move (e.g. d8h4), the destination square (e.g. h4) or the number: ").strip().lower()
        for number, move in enumerate(candidates, start=1):
            if answer in (move.uci(), chess.square_name(move.to_square), str(number)):
                return move
        print("That isn't one of the options above.")


def main(command_line_args: list[str] | None = None) -> int:
    """Run the live move-recognition test described at the top of this file."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="where the photos and labels.yaml are saved (must not already have labels)")
    parser.add_argument("--calibration-folder", default="calibration/data")
    args = parser.parse_args(command_line_args)
    with open("config/vision.yaml") as f:
        vision_settings = yaml.safe_load(f)
    if vision_settings.get("threshold") is None:
        parser.error("set threshold in config/vision.yaml first (tools/vision_live.py slider)")
    labels_path = os.path.join(args.folder, "labels.yaml")
    if os.path.exists(labels_path):
        parser.error(f"{labels_path} already exists; pick a new folder")
    calibration = load_calibration(args.calibration_folder, vision_settings)
    with open("config/hardware.yaml") as f:
        camera_settings = yaml.safe_load(f)["camera"]
    live_camera = camera.open_camera(camera_settings)
    os.makedirs(args.folder, exist_ok=True)

    board = chess.Board()
    labels = {"kind": "game", "start_fen": board.fen(), "moves": [], "photos": []}
    seen_before_move = None  # what the camera saw after the last accepted move; None until the start is confirmed
    message = "Set up the starting position, then press space"
    figure, (diagram_axes, camera_axes) = plt.subplots(1, 2, figsize=(15, 7.5))
    keys = {"space": False}
    figure.canvas.mpl_connect("key_press_event", lambda event: keys.update(space=event.key == " "))
    plt.ion()
    print(message)

    def save_photo(photo):
        name = f"{len(labels['photos']):03d}.jpg"
        cv2.imwrite(os.path.join(args.folder, name), photo, [cv2.IMWRITE_JPEG_QUALITY, 95])
        labels["photos"].append({"file": name, "fen": board.board_fen()})
        with open(labels_path, "w") as f:  # rewritten every time, so stopping early loses nothing
            yaml.safe_dump(labels, f, sort_keys=False)

    try:
        while plt.fignum_exists(figure.number) and not board.is_game_over():
            photo = camera.take_photo(live_camera, camera_settings["flush_frames"])
            _, top_down, scores = look_at_board(photo, *calibration)
            seen = scores > vision_settings["threshold"]

            if keys["space"]:
                keys["space"] = False
                if seen_before_move is None:  # first press: the starting position must look right
                    wrong = seen != expected_board_state(board)
                    if wrong.any():
                        message = f"Start position doesn't match on: {square_names(wrong)}. Fix it, press space"
                    else:
                        seen_before_move = seen
                        save_photo(photo)
                        message = "Start position OK. Make a move (white), then press space"
                else:
                    candidates = find_human_move(board, seen_before_move, seen)
                    if not candidates:
                        changed = seen != seen_before_move
                        message = (f"No legal move fits. Squares that looked changed: {square_names(changed)}. "
                                   "Fix the board, press space")
                    else:
                        move = candidates[0] if len(candidates) == 1 else ask_which_move(board, candidates)
                        message = f"You played: {board.san(move)}  ({describe_played_move(board, move)})"
                        board.push(move)
                        labels["moves"].append(move.uci())
                        seen_before_move = seen
                        save_photo(photo)
                        message += f". Now {'white' if board.turn == chess.WHITE else 'black'} to move"
                print(message)

            picture = mark_wrong_squares(draw_board_grid(top_down, seen), seen != expected_board_state(board))
            draw_position(diagram_axes, board, message)
            camera_axes.clear()
            camera_axes.imshow(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB))
            camera_axes.axis("off")
            camera_axes.set_title("live: green = piece seen, red box = differs from the game. Space = move done")
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        live_camera.release()
    if board.is_game_over():
        print(f"Game over: {board.result()}")
    print(f"{len(labels['moves'])} moves saved in {args.folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
