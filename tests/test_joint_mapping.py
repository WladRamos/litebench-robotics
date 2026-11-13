import numpy as np

from vials_lite import spec
from vials_lite.joint_mapping import (
    JOINT_OFFSETS_MJCF_RAD,
    lerobot_to_radians,
    mjcf_radians_to_usd,
    radians_to_lerobot,
    usd_radians_to_mjcf,
)


def test_roundtrip():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.uniform(-100, 100, (50, 5)), rng.uniform(0, 100, (50, 1))], axis=1)
    np.testing.assert_allclose(radians_to_lerobot(lerobot_to_radians(x)), x, atol=1e-9)


def test_extremes_match_workshop_ranges():
    lo = lerobot_to_radians(np.array([-100, -100, -100, -100, -100, 0.0]))
    hi = lerobot_to_radians(np.array([100, 100, 100, 100, 100, 100.0]))
    np.testing.assert_allclose(np.rad2deg(lo), [-110, -100, -100, -95, -160, -10])
    np.testing.assert_allclose(np.rad2deg(hi), [110, 100, 90, 95, 160, 100])


def test_initial_pose_is_inside_range():
    raw = radians_to_lerobot(np.array(spec.INITIAL_JOINT_POS))
    assert np.all(raw[:5] >= -100) and np.all(raw[:5] <= 100)
    assert 0 <= raw[5] <= 100


def test_mjcf_offset_matches_menagerie_wrist_roll_range():
    # Faixas do so101.xml do Menagerie (4d038b3): a do wrist_roll é ±160° deslocada pelo offset;
    # as demais juntas têm zero igual ao do USD.
    usd_wrist_roll = np.deg2rad([-160.0, 160.0])
    mjcf_wrist_roll = np.array([-2.7438473, 2.8412063])
    np.testing.assert_allclose(usd_wrist_roll + JOINT_OFFSETS_MJCF_RAD[4], mjcf_wrist_roll, atol=1e-4)
    assert np.count_nonzero(JOINT_OFFSETS_MJCF_RAD) == 1
    q = lerobot_to_radians(np.array([10.0, -20.0, 30.0, -40.0, 50.0, 60.0]))
    np.testing.assert_allclose(mjcf_radians_to_usd(usd_radians_to_mjcf(q)), q)
