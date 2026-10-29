"""Makes pretend camera photos of the board (and of a checkerboard) for testing the vision code without hardware.

Everything goes to data/synthetic/, with the true answers saved next to the photos (truth.yaml, labels.yaml).
The pretend camera copies the real one: 2592x1944, ~106 degree fisheye, 60 cm above the board, robot side at the bottom.
These numbers describe the pretend camera only; they are not bench values and are never read by the real pipeline.
"""
import argparse
import os
import random

import chess
import cv2
import numpy as np
import yaml

IMAGE_SIZE = (2592, 1944)                          # width, height in pixels
FOCAL_LENGTH_PX = 1751.0                           # 53 degrees off-centre lands on the image corner
LENS_CENTRE_PX = (1304.0, 966.0)                   # slightly off the image centre, like a real lens
LENS_BEND = [0.03, -0.01, 0.002, 0.0]              # OpenCV fisheye distortion coefficients k1..k4
CAMERA_POSITION_M = (0.1425, 0.1125, 0.60)         # board frame (docs/INTERFACES.md 2.6): above the board centre
CAMERA_TILT_DEG = (2.0, -1.5, 1.0)                 # small tilts so the board is not perfectly square in the photo

PLAYING_AREA_M = 0.265
BORDER_M = 0.0175                                  # wooden edge around the playing area (30 cm board in total)
TEXTURE_M_PER_PX = 0.00025                         # resolution of the drawn board picture
LIGHT_SQUARE_BGR = (150, 192, 222)
DARK_SQUARE_BGR = (45, 75, 125)
BORDER_BGR = (40, 62, 105)
WHITE_PIECE_BGR = (155, 196, 226)                  # deliberately close to the light squares, like the real set
BLACK_PIECE_BGR = (42, 70, 118)                    # deliberately close to the dark squares
PIECE_HEIGHT_M = {"pawn": 0.028, "knight": 0.040, "bishop": 0.042, "rook": 0.033, "queen": 0.046, "king": 0.049}
PIECE_BASE_RADIUS_M = 0.010
PIECE_TOP_RADIUS_M = 0.006
LIGHT_DIRECTION = np.array([-0.3, -0.4, 1.0]) / np.linalg.norm([-0.3, -0.4, 1.0])  # towards the lamp

CHECKERBOARD_INNER_CORNERS = (9, 6)                # columns, rows
CHECKERBOARD_SQUARE_M = 0.025
CHECKERBOARD_PHOTOS = 20
RANDOM_POSITION_PHOTOS = 20
GAME_OPENING = "e4 Nf6 e5 d5 exd6 Nc6 dxc7 Qxc7 Nf3 b6 Bc4 Bb7 O-O O-O-O".split()  # en passant and both castles
GAME_RANDOM_PLIES = 26


def camera_rotation() -> np.ndarray:
    """Rotation that turns board-frame directions into camera directions (camera looks down, rank 1 at the bottom)."""
    looking_down = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)  # rows: camera right, down, forward
    tilt, _ = cv2.Rodrigues(np.radians(CAMERA_TILT_DEG))
    return tilt @ looking_down


def camera_matrix() -> np.ndarray:
    return np.array([[FOCAL_LENGTH_PX, 0, LENS_CENTRE_PX[0]], [0, FOCAL_LENGTH_PX, LENS_CENTRE_PX[1]], [0, 0, 1]])


def project_to_photo(points_on_board: np.ndarray) -> np.ndarray:
    """Board-frame points in metres (N, 3) -> pixel positions (N, 2) in the raw (bent) photo."""
    rotation = camera_rotation()
    turn_vector, _ = cv2.Rodrigues(rotation)
    shift = -rotation @ np.array(CAMERA_POSITION_M)
    pixels, _ = cv2.fisheye.projectPoints(points_on_board.reshape(-1, 1, 3).astype(np.float64), turn_vector, shift,
                                          camera_matrix(), np.array(LENS_BEND))
    return pixels.reshape(-1, 2)


def viewing_directions() -> np.ndarray:
    """For every photo pixel, the direction it looks in, in camera axes, scaled so forward = 1. Shape (H, W, 3)."""
    width, height = IMAGE_SIZE
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    pixels = np.stack([u, v], axis=-1).reshape(-1, 1, 2)
    straight = cv2.fisheye.undistortPoints(pixels, camera_matrix(), np.array(LENS_BEND)).reshape(height, width, 2)
    return np.concatenate([straight, np.ones((height, width, 1), np.float32)], axis=-1)


def where_pixels_hit_table(directions: np.ndarray) -> np.ndarray:
    """For every pixel, the board-frame (x, y) where its line of sight meets the table surface (z = 0)."""
    directions_on_board = directions @ camera_rotation().astype(np.float32)  # camera -> board axes
    distance = -CAMERA_POSITION_M[2] / directions_on_board[..., 2]
    x = CAMERA_POSITION_M[0] + distance * directions_on_board[..., 0]
    y = CAMERA_POSITION_M[1] + distance * directions_on_board[..., 1]
    return np.stack([x, y], axis=-1)


def draw_board_picture(random_numbers: np.random.Generator) -> np.ndarray:
    """Top-down picture of the whole wooden board with grain and the hinge seam. Row index grows with board y."""
    side_px = round((PLAYING_AREA_M + 2 * BORDER_M) / TEXTURE_M_PER_PX)
    picture = np.empty((side_px, side_px, 3), np.float32)
    picture[:] = BORDER_BGR
    square_px = PLAYING_AREA_M / 8 / TEXTURE_M_PER_PX
    border_px = BORDER_M / TEXTURE_M_PER_PX
    rows, columns = np.mgrid[0:side_px, 0:side_px]
    for rank in range(8):
        for file in range(8):
            is_light = (rank + file) % 2 == 1  # a1 (rank 0, file 0) is dark
            top, left = border_px + rank * square_px, border_px + file * square_px
            inside = (rows >= top) & (rows < top + square_px) & (columns >= left) & (columns < left + square_px)
            grain = 6 * np.sin(rows[inside] * 0.9 + random_numbers.uniform(0, 6.3) + 0.3 * np.sin(columns[inside] * 0.05))
            picture[inside] = np.array(LIGHT_SQUARE_BGR if is_light else DARK_SQUARE_BGR) + grain[:, None]
    seam_column = round(border_px + 4 * square_px)  # the folding board's hinge, between the d and e files
    picture[:, seam_column - 1:seam_column + 2] *= 0.55
    return picture


def table_colour(points_on_table: np.ndarray) -> np.ndarray:
    """Green, striped tablecloth colour for each board-frame (x, y) point."""
    x, y = points_on_table[..., 0], points_on_table[..., 1]
    stripes = 25 * np.sin(y * 180 + 3 * np.sin(x * 20))
    return np.stack([90 + 0.3 * stripes, 150 + stripes, 110 + 0.5 * stripes], axis=-1)


def draw_piece(photo: np.ndarray, centre_x: float, centre_y: float, height: float, colour) -> None:
    """Draw one piece as a shaded, slightly narrowing cylinder, so tall pieces lean outward like in the real photos."""
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    bottom = np.column_stack([centre_x + PIECE_BASE_RADIUS_M * ring[:, 0], centre_y + PIECE_BASE_RADIUS_M * ring[:, 1],
                              np.zeros(len(angles))])
    top = np.column_stack([centre_x + PIECE_TOP_RADIUS_M * ring[:, 0], centre_y + PIECE_TOP_RADIUS_M * ring[:, 1],
                           np.full(len(angles), height)])
    pixels = project_to_photo(np.vstack([bottom, top]))
    bottom_px, top_px = pixels[:len(angles)], pixels[len(angles):]
    slope = (PIECE_BASE_RADIUS_M - PIECE_TOP_RADIUS_M) / height
    for i in range(len(angles)):
        j = (i + 1) % len(angles)
        middle_angle = angles[i] + np.pi / len(angles)
        outward = np.array([np.cos(middle_angle), np.sin(middle_angle), slope])
        towards_camera = np.array(CAMERA_POSITION_M) - bottom[i]
        if outward @ towards_camera <= 0:  # the far side of the piece is hidden
            continue
        brightness = 0.7 + 0.4 * max(0.0, outward @ LIGHT_DIRECTION / np.linalg.norm(outward))
        side = np.array([bottom_px[i], bottom_px[j], top_px[j], top_px[i]])
        fill(photo, side, np.clip(np.array(colour) * brightness, 0, 255))
    fill(photo, top_px, np.clip(np.array(colour) * (0.7 + 0.4 * LIGHT_DIRECTION[2]), 0, 255))


def fill(photo: np.ndarray, corner_pixels: np.ndarray, colour) -> None:
    fixed_point = np.round(corner_pixels * 16).astype(np.int32)  # shift=4 gives sub-pixel smooth edges
    cv2.fillPoly(photo, [fixed_point], [float(c) for c in colour], lineType=cv2.LINE_AA, shift=4)


def photo_of_position(board: chess.Board, scene: dict, random_numbers: np.random.Generator) -> np.ndarray:
    """Render one raw camera photo of the given position (an empty chess.Board gives the empty-board photo)."""
    square_m = PLAYING_AREA_M / 8
    pieces = []
    for square, piece in board.piece_map().items():
        nudge = random_numbers.uniform(-0.003, 0.003, 2)  # people never place pieces exactly in the centre
        x = (chess.square_file(square) + 0.5) * square_m + nudge[0]
        y = (chess.square_rank(square) + 0.5) * square_m + nudge[1]
        colour = WHITE_PIECE_BGR if piece.color == chess.WHITE else BLACK_PIECE_BGR
        colour = np.array(colour) + random_numbers.uniform(-6, 6, 3)
        pieces.append((x, y, PIECE_HEIGHT_M[chess.piece_name(piece.piece_type)], colour))

    board_picture = scene["board_picture"].copy()
    shadows = np.zeros(board_picture.shape[:2], np.float32)
    for x, y, height, _ in pieces:  # shadow falls away from the lamp
        shadow_x = x - 0.5 * height * LIGHT_DIRECTION[0] / LIGHT_DIRECTION[2]
        shadow_y = y - 0.5 * height * LIGHT_DIRECTION[1] / LIGHT_DIRECTION[2]
        centre_px = (round((shadow_x + BORDER_M) / TEXTURE_M_PER_PX), round((shadow_y + BORDER_M) / TEXTURE_M_PER_PX))
        cv2.circle(shadows, centre_px, round(1.2 * PIECE_BASE_RADIUS_M / TEXTURE_M_PER_PX), 1.0, -1)
    shadows = cv2.GaussianBlur(shadows, (0, 0), 12)
    board_picture *= (1 - 0.3 * shadows)[..., None]

    photo = cv2.remap(board_picture, scene["texture_columns"], scene["texture_rows"], cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=(-1, -1, -1))
    off_board = photo[..., 0] < 0
    photo[off_board] = scene["table"][off_board]
    distance_to_camera = lambda piece: -np.hypot(piece[0] - CAMERA_POSITION_M[0], piece[1] - CAMERA_POSITION_M[1])
    for x, y, height, colour in sorted(pieces, key=distance_to_camera):  # far pieces first, near ones drawn over them
        draw_piece(photo, x, y, height, colour)
    return finish_photo(photo + scene["glare"], random_numbers)


def finish_photo(photo: np.ndarray, random_numbers: np.random.Generator) -> np.ndarray:
    """Soften like a real lens, add small light changes and sensor noise, and convert to 8-bit."""
    photo = cv2.GaussianBlur(photo, (0, 0), 0.8) * random_numbers.uniform(0.98, 1.02)
    photo += random_numbers.normal(0, 2.0, photo.shape).astype(np.float32)
    return np.clip(photo, 0, 255).astype(np.uint8)


def photo_of_checkerboard(directions: np.ndarray, random_numbers: np.random.Generator) -> np.ndarray:
    """Render a printed checkerboard held at a random place and angle, fully inside the photo."""
    columns, rows = CHECKERBOARD_INNER_CORNERS
    half_width, half_height = (columns + 1) / 2 * CHECKERBOARD_SQUARE_M, (rows + 1) / 2 * CHECKERBOARD_SQUARE_M
    margin = 0.015
    width, height = IMAGE_SIZE
    while True:
        aim_pixel = np.array([[[random_numbers.uniform(0.12, 0.88) * width, random_numbers.uniform(0.12, 0.88) * height]]])
        aim = cv2.fisheye.undistortPoints(aim_pixel, camera_matrix(), np.array(LENS_BEND)).reshape(2)
        centre = np.array([aim[0], aim[1], 1.0]) * random_numbers.uniform(0.3, 0.55)
        tilt = random_numbers.normal(0, 1, 3)
        tilt *= np.radians(random_numbers.uniform(0, 40)) / np.linalg.norm(tilt)
        rotation = cv2.Rodrigues(tilt)[0] @ cv2.Rodrigues(np.array([0, 0, random_numbers.uniform(-0.4, 0.4)]))[0]
        outer = np.array([[sx * (half_width + margin), sy * (half_height + margin), 0] for sx in (-1, 1) for sy in (-1, 1)])
        corners_in_camera = outer @ rotation.T + centre
        corners_px, _ = cv2.fisheye.projectPoints(corners_in_camera.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                                  camera_matrix(), np.array(LENS_BEND))
        corners_px = corners_px.reshape(-1, 2)
        if (corners_in_camera[:, 2] > 0.1).all() and (corners_px > 20).all() and (corners_px < [width - 20, height - 20]).all():
            break

    normal = rotation[:, 2]
    distance = (normal @ centre) / (directions @ normal.astype(np.float32))
    on_target = directions * distance[..., None] - centre.astype(np.float32)
    a, b = on_target @ rotation[:, 0].astype(np.float32), on_target @ rotation[:, 1].astype(np.float32)
    background = 100 + 20 * np.sin(directions[..., 0] * 3)[..., None] * np.ones(3, np.float32)
    on_paper = (np.abs(a) < half_width + margin) & (np.abs(b) < half_height + margin) & (distance > 0)
    on_squares = (np.abs(a) < half_width) & (np.abs(b) < half_height)
    is_black = (np.floor((a + half_width) / CHECKERBOARD_SQUARE_M) + np.floor((b + half_height) / CHECKERBOARD_SQUARE_M)) % 2 == 0
    photo = background
    photo[on_paper] = 230
    photo[on_paper & on_squares & is_black] = 25
    return finish_photo(photo.astype(np.float32), random_numbers)


def random_position(random_numbers: random.Random) -> chess.Board:
    """A board with a random number of random pieces on random squares (not a legal game, just occupancy)."""
    board = chess.Board(None)
    for square in random_numbers.sample(range(64), random_numbers.randint(1, 32)):
        board.set_piece_at(square, chess.Piece(random_numbers.randint(1, 6), random_numbers.random() < 0.5))
    return board


def recorded_game(random_numbers: random.Random) -> list[chess.Move]:
    """A fixed opening (en passant, both castles) followed by random legal moves, with captures preferred."""
    board = chess.Board()
    moves = [board.push_san(san) for san in GAME_OPENING]
    for _ in range(GAME_RANDOM_PLIES):
        choices = [m for m in board.legal_moves if m.promotion in (None, chess.QUEEN)]
        if not choices:
            break
        captures = [m for m in choices if board.is_capture(m)]
        move = random_numbers.choice(captures if captures and random_numbers.random() < 0.5 else choices)
        board.push(move)
        moves.append(move)
    return moves


def save_photo(folder: str, name: str, photo: np.ndarray) -> str:
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, name), photo, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return name


def main(command_line_args: list[str] | None = None) -> int:
    """Write the whole synthetic set (checkerboards, empty board, random positions, one game) with its answers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-folder", default="data/synthetic")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(command_line_args)
    out = args.output_folder
    pixel_randomness, choice_randomness = np.random.default_rng(args.seed), random.Random(args.seed)

    print("working out where every pixel looks ...")
    directions = viewing_directions()
    points_on_table = where_pixels_hit_table(directions)
    width, height = IMAGE_SIZE
    glare_u, glare_v = project_to_photo(np.array([[0.19, 0.17, 0.0]]))[0]
    v, u = np.mgrid[0:height, 0:width]
    glare = 55 * np.exp(-((u - glare_u) ** 2 + (v - glare_v) ** 2) / (2 * 70.0 ** 2))  # a lamp reflection that never moves
    scene = {
        "board_picture": draw_board_picture(pixel_randomness),
        "texture_columns": ((points_on_table[..., 0] + BORDER_M) / TEXTURE_M_PER_PX).astype(np.float32),
        "texture_rows": ((points_on_table[..., 1] + BORDER_M) / TEXTURE_M_PER_PX).astype(np.float32),
        "table": table_colour(points_on_table).astype(np.float32),
        "glare": glare[..., None].astype(np.float32),
    }

    for i in range(CHECKERBOARD_PHOTOS):
        print(f"checkerboard {i + 1}/{CHECKERBOARD_PHOTOS}")
        save_photo(os.path.join(out, "checkerboard"), f"{i:02d}.jpg", photo_of_checkerboard(directions, pixel_randomness))

    save_photo(out, "empty_board.jpg", photo_of_position(chess.Board(None), scene, pixel_randomness))

    position_labels = []
    for i in range(RANDOM_POSITION_PHOTOS):
        board = chess.Board() if i == 0 else random_position(choice_randomness)
        print(f"position {i + 1}/{RANDOM_POSITION_PHOTOS}: {board.board_fen()}")
        name = save_photo(os.path.join(out, "positions"), f"{i:02d}.jpg", photo_of_position(board, scene, pixel_randomness))
        position_labels.append({"file": name, "fen": board.board_fen()})
    with open(os.path.join(out, "positions", "labels.yaml"), "w") as f:
        yaml.safe_dump({"photos": position_labels}, f, sort_keys=False)

    board = chess.Board()
    moves = recorded_game(choice_randomness)
    game_labels = []
    for i in range(len(moves) + 1):
        print(f"game photo {i}/{len(moves)}")
        name = save_photo(os.path.join(out, "game"), f"{i:03d}.jpg", photo_of_position(board, scene, pixel_randomness))
        game_labels.append({"file": name, "fen": board.board_fen()})
        if i < len(moves):
            board.push(moves[i])
    with open(os.path.join(out, "game", "labels.yaml"), "w") as f:
        yaml.safe_dump({"start_fen": chess.STARTING_FEN, "moves": [m.uci() for m in moves],
                        "photos": game_labels}, f, sort_keys=False)

    corners = np.array([[0, 0, 0], [PLAYING_AREA_M, 0, 0], [PLAYING_AREA_M, PLAYING_AREA_M, 0], [0, PLAYING_AREA_M, 0]])
    truth = {
        "note": "synthetic data from tools/make_synthetic.py; never use as real calibration",
        "lens_model": "fisheye",
        "image_size": list(IMAGE_SIZE),
        "camera_matrix": camera_matrix().tolist(),
        "distortion": LENS_BEND,
        "camera_position_m": list(CAMERA_POSITION_M),
        "corner_pixels_raw": project_to_photo(corners).tolist(),  # a1, h1, h8, a8 outer corners, in the raw photo
        "checkerboard": {"inner_corners": list(CHECKERBOARD_INNER_CORNERS), "square_mm": CHECKERBOARD_SQUARE_M * 1000},
        "piece_height_m": PIECE_HEIGHT_M,
    }
    with open(os.path.join(out, "truth.yaml"), "w") as f:
        yaml.safe_dump(truth, f, sort_keys=False)
    print(f"done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
