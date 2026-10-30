"""Collects labelled board photos (the photo plus which squares really have pieces) and checks the vision on them.

Collect (needs the camera; arm out of view):
  python tools/collect_states.py empty  data/images/states/empty   [--count 5]
  python tools/collect_states.py random data/images/states/random  [--count 25] [--seed 1]
  python tools/collect_states.py game   data/images/states/game    [--seed 1]
  Set up what the left panel shows, keep hands out of view, press space to save and go on. Red boxes on the
  right = squares the vision currently gets wrong. Close the window to stop; what is saved so far is kept.
Check (no camera):
  python tools/collect_states.py check data/images/states/random
"""
import argparse
import os
import random
import sys

import chess
import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Run as a script from the repo root, so make the project folders importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chess_logic.move_resolver import expected_board_state, find_human_move  # noqa: E402
from tools.vision_live import draw_board_grid, look_at_board  # noqa: E402
from vision import board_view, camera  # noqa: E402

# A fixed opening that contains en passant (3. exd6) and both castles, then random moves.
GAME_OPENING = "e4 Nf6 e5 d5 exd6 Nc6 dxc7 Qxc7 Nf3 b6 Bc4 Bb7 O-O O-O-O".split()
GAME_RANDOM_MOVES = 26
FULL_SET = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1, chess.KING: 1}
SYMBOLS = {"P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
           "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"}


def random_position(random_numbers: random.Random) -> chess.Board:
    """Random pieces (never more than one real set has) on random squares, many on a square of their own colour.

    Pieces the same colour as their square are the hardest for the camera, so most pieces are put there on purpose.
    """
    pieces = [chess.Piece(piece_type, colour) for colour in chess.COLORS
              for piece_type, count in FULL_SET.items() for _ in range(count)]
    board = chess.Board(None)
    free_squares = list(chess.SQUARES)
    for piece in random_numbers.sample(pieces, random_numbers.randint(4, 24)):
        light_square_wanted = piece.color == chess.WHITE
        matching = [s for s in free_squares if ((chess.square_file(s) + chess.square_rank(s)) % 2 == 1) == light_square_wanted]
        square = random_numbers.choice(matching if matching and random_numbers.random() < 0.7 else free_squares)
        board.set_piece_at(square, piece)
        free_squares.remove(square)
    return board


def describe_move(board: chess.Board, move: chess.Move) -> str:
    """Plain instructions for making one move by hand, e.g. 'white knight g1 -> f3'."""
    colour = "white" if board.turn == chess.WHITE else "black"
    start, end = chess.square_name(move.from_square), chess.square_name(move.to_square)
    piece_name = chess.piece_name(board.piece_type_at(move.from_square))
    if board.is_castling(move):
        rook_start, rook_end = ("h", "f") if chess.square_file(move.to_square) == 6 else ("a", "d")
        rank = start[1]
        return f"{colour} castles: king {start} -> {end}, rook {rook_start}{rank} -> {rook_end}{rank}"
    text = f"{colour} {piece_name} {start} -> {end}"
    if board.is_en_passant(move):
        taken_square = chess.square_name(chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square)))
        text += f" (en passant: remove the pawn on {taken_square})"
    elif board.is_capture(move):
        text += f" (remove the {chess.piece_name(board.piece_type_at(move.to_square))} on {end} first)"
    if move.promotion:
        text += ", then replace the pawn with a queen (a captured one, or any stand-in piece: there is only one set)"
    return text


def plan_targets(kind: str, count: int, seed: int) -> tuple[list[tuple[chess.Board, str]], dict]:
    """The positions to photograph, each with the instruction shown, plus extra label fields (the game's moves)."""
    random_numbers = random.Random(seed)
    if kind == "empty":
        return [(chess.Board(None), f"clear the board (photo {i + 1}/{count})") for i in range(count)], {}
    if kind == "random":
        targets = [(chess.Board(), "starting position")]
        targets += [(random_position(random_numbers), f"random position {i + 2}/{count}") for i in range(count - 1)]
        return targets, {}

    board = chess.Board()
    targets = [(board.copy(), "set up the starting position")]
    moves = []
    for move_number in range(len(GAME_OPENING) + GAME_RANDOM_MOVES):
        if move_number < len(GAME_OPENING):
            move = board.parse_san(GAME_OPENING[move_number])
        else:
            choices = [m for m in board.legal_moves if m.promotion in (None, chess.QUEEN)]  # vision assumes queens
            if not choices:
                break
            captures = [m for m in choices if board.is_capture(m)]
            move = random_numbers.choice(captures if captures and random_numbers.random() < 0.5 else choices)
        instruction = f"move {move_number + 1}: {describe_move(board, move)}"
        board.push(move)
        moves.append(move.uci())
        targets.append((board.copy(), instruction))
    return targets, {"start_fen": chess.STARTING_FEN, "moves": moves}


def load_calibration(calibration_folder: str, vision_settings: dict):
    """Lens, board and empty-board reference, in the order look_at_board expects."""
    return (camera.load_lens_calibration(os.path.join(calibration_folder, "camera_intrinsics.yaml")),
            board_view.load_board_calibration(os.path.join(calibration_folder, "board.yaml")),
            cv2.imread(os.path.join(calibration_folder, "empty_board.png")),
            vision_settings)


def draw_position(axes, board: chess.Board, title: str) -> None:
    """Draw a chessboard diagram (rank 1 at the bottom, like the top-down view) with piece symbols."""
    axes.clear()
    for rank in range(8):
        for file in range(8):
            is_light = (file + rank) % 2 == 1
            axes.add_patch(plt.Rectangle((file, rank), 1, 1, color="#e8d0aa" if is_light else "#a07040"))
            piece = board.piece_at(chess.square(file, rank))
            if piece:
                axes.text(file + 0.5, rank + 0.45, SYMBOLS[piece.symbol()], ha="center", va="center", fontsize=26)
    axes.set_xlim(0, 8)
    axes.set_ylim(0, 8)
    axes.set_aspect("equal")
    axes.set_xticks([f + 0.5 for f in range(8)], list("abcdefgh"))
    axes.set_yticks([r + 0.5 for r in range(8)], [str(r + 1) for r in range(8)])
    axes.set_title(title, fontsize=10)


def mark_wrong_squares(top_down_picture: np.ndarray, wrong: np.ndarray) -> np.ndarray:
    """Draw a thick red box on every square where the vision disagrees with the position."""
    square_px = top_down_picture.shape[0] / 8
    for rank_index, file_index in zip(*np.nonzero(wrong)):
        left, top = round(file_index * square_px), round((7 - rank_index) * square_px)
        cv2.rectangle(top_down_picture, (left + 2, top + 2), (round(left + square_px) - 2, round(top + square_px) - 2),
                      (0, 0, 255), 3)
    return top_down_picture


def collect(kind: str, folder: str, count: int, seed: int, calibration_folder: str, vision_settings: dict) -> int:
    """Walk through the planned positions, saving a raw photo for each when space is pressed. Returns photos saved."""
    labels_path = os.path.join(folder, "labels.yaml")
    if os.path.exists(labels_path):
        print(f"{labels_path} already exists; pick a new folder so earlier photos are not mixed up.")
        return 0
    targets, extra_labels = plan_targets(kind, count, seed)
    calibration = load_calibration(calibration_folder, vision_settings)
    threshold = vision_settings.get("threshold")
    with open("config/hardware.yaml") as f:
        camera_settings = yaml.safe_load(f)["camera"]
    live_camera = camera.open_camera(camera_settings)
    os.makedirs(folder, exist_ok=True)

    figure, (diagram_axes, camera_axes) = plt.subplots(1, 2, figsize=(15, 7.5))
    keys = {"space": False}
    figure.canvas.mpl_connect("key_press_event", lambda event: keys.update(space=event.key == " "))
    labels = {"kind": kind, **extra_labels, "photos": []}
    plt.ion()
    try:
        while len(labels["photos"]) < len(targets) and plt.fignum_exists(figure.number):
            board, instruction = targets[len(labels["photos"])]
            photo = camera.take_photo(live_camera, camera_settings["flush_frames"])
            _, top_down, scores = look_at_board(photo, *calibration)
            should_have_piece = expected_board_state(board)
            seen = None if threshold is None else scores > threshold
            picture = draw_board_grid(top_down, seen)
            if seen is not None:
                picture = mark_wrong_squares(picture, seen != should_have_piece)

            if keys["space"]:
                keys["space"] = False
                name = f"{len(labels['photos']):03d}.jpg"
                cv2.imwrite(os.path.join(folder, name), photo, [cv2.IMWRITE_JPEG_QUALITY, 95])
                labels["photos"].append({"file": name, "fen": board.board_fen()})
                with open(labels_path, "w") as f:  # rewritten after every photo, so stopping early loses nothing
                    yaml.safe_dump(labels, f, sort_keys=False)
                wrong_count = 0 if seen is None else int((seen != should_have_piece).sum())
                print(f"saved {name}: {instruction}" + (f"  ({wrong_count} squares wrong)" if wrong_count else ""))
                continue

            draw_position(diagram_axes, board, f"{len(labels['photos']) + 1}/{len(targets)}  {instruction}")
            camera_axes.clear()
            camera_axes.imshow(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB))
            camera_axes.axis("off")
            camera_axes.set_title("live: green = piece seen, red box = vision disagrees. Space = save")
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        live_camera.release()
    print(f"{len(labels['photos'])} of {len(targets)} photos saved in {folder}")
    return len(labels["photos"])


def check(folder: str, calibration_folder: str, vision_settings: dict) -> int:
    """Score every saved photo against its label and print where the vision is wrong. Returns the number of mistakes."""
    with open(os.path.join(folder, "labels.yaml")) as f:
        labels = yaml.safe_load(f)
    calibration = load_calibration(calibration_folder, vision_settings)
    threshold = vision_settings.get("threshold")
    empty_scores, occupied_scores, seen_per_photo, mistakes = [], [], [], 0
    for entry in labels["photos"]:
        _, _, scores = look_at_board(cv2.imread(os.path.join(folder, entry["file"])), *calibration)
        should_have_piece = expected_board_state(chess.Board(entry["fen"] + " w - - 0 1"))
        empty_scores += [(scores[r, f], entry["file"], chess.square_name(chess.square(f, r)))
                         for r, f in zip(*np.nonzero(~should_have_piece))]
        occupied_scores += [(scores[r, f], entry["file"], chess.square_name(chess.square(f, r)))
                            for r, f in zip(*np.nonzero(should_have_piece))]
        if threshold is None:
            continue
        seen = scores > threshold
        seen_per_photo.append(seen)
        wrong = [chess.square_name(chess.square(f, r)) for r, f in zip(*np.nonzero(seen != should_have_piece))]
        mistakes += len(wrong)
        print(f"{entry['file']}: " + ("all 64 squares right" if not wrong else f"WRONG {' '.join(wrong)}"))

    if empty_scores:
        highest_empty = max(empty_scores)
        print(f"highest empty-square score:    {highest_empty[0]:6.1f}  ({highest_empty[1]} {highest_empty[2]})")
    if occupied_scores:
        lowest_occupied = min(occupied_scores)
        print(f"lowest occupied-square score:  {lowest_occupied[0]:6.1f}  ({lowest_occupied[1]} {lowest_occupied[2]})")
    if empty_scores and occupied_scores:
        if lowest_occupied[0] > highest_empty[0]:
            print(f"every threshold between them works; middle = {(highest_empty[0] + lowest_occupied[0]) / 2:.1f}"
                  f" (current: {threshold})")
        else:
            print("NO threshold separates them: some empty square scores higher than some piece.")

    if "moves" in labels and seen_per_photo:
        board = chess.Board(labels["start_fen"])
        found = 0
        photographed_moves = labels["moves"][:len(seen_per_photo) - 1]  # a stopped game has fewer photos than moves
        for move_index, uci in enumerate(photographed_moves):
            move = chess.Move.from_uci(uci)
            candidates = find_human_move(board, seen_per_photo[move_index], seen_per_photo[move_index + 1])
            if move in candidates:
                found += 1
                if len(candidates) > 1:
                    print(f"move {move_index + 1} {uci}: found, ambiguous among {[m.uci() for m in candidates]}")
            else:
                print(f"move {move_index + 1} {uci}: NOT found (vision gave {[m.uci() for m in candidates]})")
            board.push(move)
        print(f"game: {found}/{len(photographed_moves)} photographed moves recovered by find_human_move"
              + (f" ({len(labels['moves']) - len(photographed_moves)} planned moves not photographed)"
                 if len(photographed_moves) < len(labels["moves"]) else ""))
    return mistakes


def main(command_line_args: list[str] | None = None) -> int:
    """Run the collection or the check described at the top of this file."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=["empty", "random", "game", "check"])
    parser.add_argument("folder")
    parser.add_argument("--count", type=int, help="how many photos (default: empty 5, random 25)")
    parser.add_argument("--seed", type=int, default=1, help="change it for a different set of random positions/moves")
    parser.add_argument("--calibration-folder", default="calibration/data")
    args = parser.parse_args(command_line_args)
    with open("config/vision.yaml") as f:
        vision_settings = yaml.safe_load(f)
    if args.kind == "check":
        return 1 if check(args.folder, args.calibration_folder, vision_settings) else 0
    count = args.count or {"empty": 5, "random": 25, "game": 0}[args.kind]
    collect(args.kind, args.folder, count, args.seed, args.calibration_folder, vision_settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
