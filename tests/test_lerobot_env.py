"""Integração com o LeRobot. Pulado sem o extra ``lerobot`` ou sem o Menagerie."""

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("lerobot")

from vials_lite.scene import default_menagerie_dir  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (default_menagerie_dir() / "robotstudio_so101" / "so101.xml").is_file(),
    reason="MuJoCo Menagerie ausente (ver assets/README.md)",
)


def test_policy_features_match_dataset_keys():
    from lerobot.envs.utils import env_to_policy_features

    from vials_lite.lerobot_env import VialsLiteEnvConfig

    feats = env_to_policy_features(VialsLiteEnvConfig())
    assert feats["observation.state"].shape == (6,)
    assert feats["action"].shape == (6,)
    for cam in ("ego", "external_D455"):
        assert feats[f"observation.images.{cam}"].shape == (3, 480, 640)


def test_vector_env_runs_and_preprocesses():
    from lerobot.envs.utils import preprocess_observation

    from vials_lite.lerobot_env import VialsLiteEnvConfig, make_env

    vec = make_env(n_envs=2, cfg=VialsLiteEnvConfig(variant="dr-eval", task="pick"))["vials_lite"][0]
    try:
        obs, info = vec.reset(seed=[0, 1])
        batch = preprocess_observation(obs)
        assert batch["observation.images.external_D455"].shape == (2, 3, 480, 640)
        assert batch["observation.state"].shape == (2, 6)
        assert list(vec.call("_max_episode_steps")) == [450, 450]
        assert list(vec.call("task_description")) == ["pick", "pick"]  # --env.task chega ao ambiente
        assert set(info["privileged"]["robot_color"]) <= {"orange", "teal", "white", "black"}
        for _ in range(3):
            obs, _, _, _, info = vec.step(vec.action_space.sample())
        assert info["is_success"].shape == (2,)
        assert not np.any(info["is_success"])
    finally:
        vec.close()


def test_config_rejects_unknown_variant():
    from vials_lite.lerobot_env import VialsLiteEnvConfig

    with pytest.raises(ValueError):
        VialsLiteEnvConfig(variant="foo")
    with pytest.raises(ValueError):
        VialsLiteEnvConfig(layout="foo")
    assert VialsLiteEnvConfig(layout="dataset").gym_kwargs["layout"] == "dataset"
    kw = VialsLiteEnvConfig(init_pose="dataset", external_cam="fixed").gym_kwargs
    assert kw["init_pose"] == "dataset" and kw["external_cam"] == "fixed"
    with pytest.raises(ValueError):
        VialsLiteEnvConfig(external_cam="foo")
