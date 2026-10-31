"""pytest: the arm model reaches every square and gives back the same point when run in reverse."""
import os

import chess
import numpy as np
import pytest
import yaml

pytest.importorskip("placo")
from control import kinematics  # noqa: E402

CALIBRATION_FOLDER = "calibration/data"
TEST_WRIST_ROLL = 90.0  # used where arm.yaml's wrist_roll_deg may still be unset


@pytest.fixture(scope="module")
def arm_settings():
    with open("config/arm.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def board_settings():
    with open("config/board.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def made_up_calibration(tmp_path, arm_settings):
    """A calibration folder with the board 8 cm in front of the base, centred, and a simple park pose."""
    board_to_robot = np.eye(4)
    board_to_robot[:3, :3] = kinematics.BOARD_FACING_ROBOT
    board_to_robot[:3, 3] = [0.08, 0.1325, 0.0]
    with open(tmp_path / "arm_board.yaml", "w") as f:
        yaml.safe_dump({"board_to_robot_transform": board_to_robot.tolist(), "touch_height_m": 0.02,
                        "samples": [], "square_corrections": []}, f)
    park_pose = {"shoulder_pan": 0.0, "shoulder_lift": -30.0, "elbow_flex": 30.0, "wrist_flex": 60.0,
                 "wrist_roll": TEST_WRIST_ROLL, "gripper": 0.0}
    with open(tmp_path / "park_pose.yaml", "w") as f:
        yaml.safe_dump(park_pose, f)
    return kinematics.load_arm_calibration(str(tmp_path), arm_settings, allow_rough_defaults=False)


def test_square_centres(board_settings):
    size = board_settings["square_size_m"]
    assert kinematics.square_center_on_board(chess.A1, board_settings) == pytest.approx((size / 2, size / 2))
    assert kinematics.square_center_on_board(chess.E2, board_settings) == pytest.approx((0.1490625, 0.0496875))
    assert kinematics.square_center_on_board(chess.H8, board_settings) == pytest.approx((7.5 * size, 7.5 * size))
    for not_a_square in (-1, 64):
        with pytest.raises(ValueError):
            kinematics.square_center_on_board(not_a_square, board_settings)


def test_tilt_is_steep_near_and_flat_far(arm_settings):
    settings = dict(arm_settings, reach_near_m=0.1, reach_far_m=0.3, pitch_near_deg=80.0, pitch_far_deg=40.0)
    assert kinematics.gripper_tilt_for(np.array([0.05, 0.0, 0.0]), settings) == pytest.approx(80.0)
    assert kinematics.gripper_tilt_for(np.array([0.0, 0.2, 0.0]), settings) == pytest.approx(60.0)
    assert kinematics.gripper_tilt_for(np.array([0.5, 0.0, 0.0]), settings) == pytest.approx(40.0)
    with pytest.raises(ValueError):
        kinematics.gripper_tilt_for(np.array([0.2, 0.0, 0.0]), dict(settings, pitch_far_deg=None))


def poses_over_the_board(settings, calibration, how_many):
    """Random arm poses whose tip is over the board area, jaws tilted down and pointing away from the base."""
    random_numbers = np.random.default_rng(seed=1)
    poses = []
    while len(poses) < how_many:
        pose = {"shoulder_pan": random_numbers.uniform(-60, 60), "shoulder_lift": random_numbers.uniform(-100, 100),
                "elbow_flex": random_numbers.uniform(-97, 97), "wrist_flex": random_numbers.uniform(-95, 95),
                "wrist_roll": TEST_WRIST_ROLL}
        point, tilt = kinematics.gripper_position_for(pose, settings, calibration)
        frame_pose = calibration["arm_model"].forward_kinematics(np.array([pose[j] for j in kinematics.BODY_JOINTS]))
        pointing_outward = frame_pose[:2, 2] @ point[:2] > 0
        over_board = 0.08 < point[0] < 0.42 and abs(point[1]) < 0.2 and 0.01 < point[2] < 0.15
        if over_board and 20 < tilt < 89 and pointing_outward:
            poses.append(pose)
    return poses


def test_joint_angles_round_trip(arm_settings, made_up_calibration):
    """Poses made by running the arm model forward must be found again by the reverse search."""
    settings = dict(arm_settings, wrist_roll_deg=TEST_WRIST_ROLL)
    for pose in poses_over_the_board(settings, made_up_calibration, how_many=50):
        point, tilt = kinematics.gripper_position_for(pose, settings, made_up_calibration)
        found = kinematics.joint_angles_for(point, tilt, made_up_calibration["park_pose"], settings,
                                            made_up_calibration)
        assert found is not None, f"no solution for {pose}"
        found_point, found_tilt = kinematics.gripper_position_for(found, settings, made_up_calibration)
        assert np.linalg.norm(found_point - point) <= settings["ik_pos_tol_m"]
        assert abs(found_tilt - tilt) <= settings["ik_pitch_tol_deg"]


def test_out_of_reach_gives_none(arm_settings, made_up_calibration):
    settings = dict(arm_settings, wrist_roll_deg=TEST_WRIST_ROLL)
    one_metre_away = np.array([1.0, 0.0, 0.05])
    assert kinematics.joint_angles_for(one_metre_away, 30.0, made_up_calibration["park_pose"], settings,
                                       made_up_calibration) is None


def test_joint_limit_from_config_gives_none(arm_settings, made_up_calibration):
    settings = dict(arm_settings, wrist_roll_deg=TEST_WRIST_ROLL)
    pose = {"shoulder_pan": 20.0, "shoulder_lift": 0.0, "elbow_flex": 30.0, "wrist_flex": 60.0,
            "wrist_roll": TEST_WRIST_ROLL}
    point, tilt = kinematics.gripper_position_for(pose, settings, made_up_calibration)
    assert kinematics.joint_angles_for(point, tilt, pose, settings, made_up_calibration) is not None
    pan_limited = dict(settings, joint_limits_deg={"shoulder_pan": [-10.0, 10.0]})
    assert kinematics.joint_angles_for(point, tilt, pose, pan_limited, made_up_calibration) is None


def test_missing_joint_raises_key_error(arm_settings, made_up_calibration):
    with pytest.raises(KeyError):
        kinematics.gripper_position_for({"shoulder_pan": 0.0}, arm_settings, made_up_calibration)


def test_missing_calibration(tmp_path, arm_settings):
    with pytest.raises(FileNotFoundError):
        kinematics.load_arm_calibration(str(tmp_path), arm_settings, allow_rough_defaults=False)
    no_defaults = dict(arm_settings, nominal={"board_origin_base_m": [None, None, 0.0], "park_pose_deg": None})
    with pytest.raises(FileNotFoundError):
        kinematics.load_arm_calibration(str(tmp_path), no_defaults, allow_rough_defaults=True)


def test_rough_defaults_face_the_robot(tmp_path, arm_settings):
    """With --mock defaults, a1 is front-left of the robot and h1 front-right (docs/INTERFACES.md §2.7)."""
    park_pose = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0, "wrist_flex": 0.0,
                 "wrist_roll": 0.0, "gripper": 0.0}
    defaults = dict(arm_settings, nominal={"board_origin_base_m": [0.08, 0.1325, 0.0], "park_pose_deg": park_pose})
    calibration = kinematics.load_arm_calibration(str(tmp_path), defaults, allow_rough_defaults=True)
    a1_corner = kinematics.board_point_to_robot((0.0, 0.0), 0.0, calibration)
    h1_corner = kinematics.board_point_to_robot((0.265, 0.0), 0.0, calibration)
    a8_corner = kinematics.board_point_to_robot((0.0, 0.265), 0.0, calibration)
    assert a1_corner == pytest.approx([0.08, 0.1325, 0.0])
    assert h1_corner == pytest.approx([0.08, -0.1325, 0.0])
    assert a8_corner == pytest.approx([0.345, 0.1325, 0.0])
    assert calibration["park_pose"] == park_pose


def test_corrections_hit_measured_squares_exactly(made_up_calibration):
    calibration = dict(made_up_calibration, square_corrections=[((0.0165625, 0.0165625), (0.003, -0.002)),
                                                                 ((0.2484375, 0.2484375), (-0.001, 0.0))])
    without = kinematics.board_point_to_robot((0.0195625, 0.0145625), 0.0, made_up_calibration)
    with_correction = kinematics.board_point_to_robot((0.0165625, 0.0165625), 0.0, calibration)
    assert with_correction == pytest.approx(without)
    # Far from both, the correction is a blend: between the two measured values.
    middle_shift = kinematics.correction_at((0.1325, 0.1325), calibration["square_corrections"])
    assert -0.001 <= middle_shift[0] <= 0.003


def test_every_square_and_graveyard_reachable(arm_settings, board_settings):
    """The real setup: every square at the heights the arm uses, plus the graveyard. Lists what can't be reached."""
    if not os.path.exists(os.path.join(CALIBRATION_FOLDER, "arm_board.yaml")):
        pytest.skip("calibration/data/arm_board.yaml missing: run calibration/arm_board.py")
    schedule = ("reach_near_m", "reach_far_m", "pitch_near_deg", "pitch_far_deg", "wrist_roll_deg")
    unset = [key for key in schedule if arm_settings[key] is None]
    if unset:
        pytest.skip(f"arm.yaml still unset: {', '.join(unset)}")
    calibration = kinematics.load_arm_calibration(CALIBRATION_FOLDER, arm_settings, allow_rough_defaults=False)

    known_heights = [height for height in board_settings["piece_height_m"].values() if height is not None]
    heights = {"grip": arm_settings["grip_height_frac"] * min(known_heights),
               "hover": max(known_heights) + arm_settings["hover_height_m"]}
    if arm_settings["travel_height_m"] is not None:
        heights["travel"] = arm_settings["travel_height_m"]
    spots = {chess.square_name(square): kinematics.square_center_on_board(square, board_settings)
             for square in chess.SQUARES}
    graveyard = board_settings["graveyard_xy_m"][0]
    if None not in graveyard:
        spots["graveyard"] = tuple(graveyard)

    unreachable = []
    for name, board_x_y in spots.items():
        for height_name, height in heights.items():
            point = kinematics.board_point_to_robot(board_x_y, height, calibration)
            tilt = kinematics.gripper_tilt_for(point, arm_settings)
            if kinematics.joint_angles_for(point, tilt, calibration["park_pose"], arm_settings, calibration) is None:
                unreachable.append(f"{name} ({height_name})")
    assert not unreachable, f"can't reach {len(unreachable)}: {', '.join(unreachable)}"
