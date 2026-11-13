"""Convenção de frames do ``scripts/render_dataset.py``, com um ambiente falso."""

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_script():
    scripts = Path(__file__).parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location("render_dataset", scripts / "render_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeEnv:
    """As juntas observadas são a última ação recebida (o reset dá a pose da cena); sucesso no step 3."""

    def __init__(self):
        self.reset_args = None

    def _obs(self, value):
        img = np.full((2, 2, 3), self.n, np.uint8)
        return {"agent_pos": np.full(6, value, np.float32), "pixels": {"ego": img, "external_D455": img}}

    def reset(self, seed=None, options=None):
        self.reset_args, self.n = (seed, options), 0
        return self._obs(-1.0), {}

    def step(self, action):
        self.n += 1
        return self._obs(action[0]), 0.0, self.n >= 3, False, {}


def test_frame_t_pairs_the_observation_after_action_t_with_action_t():
    mod = _load_script()
    env = FakeEnv()
    actions = np.arange(5, dtype=np.float32)[:, None].repeat(6, axis=1)
    frames = list(mod.render_episode(env, actions, scene={"joint_pos": [0] * 6}))
    assert env.reset_args == (0, {"scene": {"joint_pos": [0] * 6}})
    assert len(frames) == 5  # um frame por ação gravada, sem parar no sucesso
    # frame 0: observação do reset com action[0]; frame t: depois de step(action[t]), com action[t]
    assert frames[0]["observation.state"][0] == -1.0
    assert [f["observation.state"][0] for f in frames[1:]] == [1.0, 2.0, 3.0, 4.0]
    np.testing.assert_array_equal(np.stack([f["action"] for f in frames]), actions)
    assert [int(f["observation.images.ego"][0, 0, 0]) for f in frames] == [0, 1, 2, 3, 4]
    assert [f["success"] for f in frames] == [False, False, False, True, True]


def test_dataset_features_drops_default_features_and_video_info():
    mod = _load_script()
    info = {"features": {
        "action": {"dtype": "float32", "shape": [6], "names": ["a"] * 6, "fps": 30},
        "observation.images.ego": {"dtype": "video", "shape": [480, 640, 3], "names": ["h", "w", "c"],
                                   "info": {"video.codec": "av1"}},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    }}
    feats = mod.dataset_features(info)
    assert set(feats) == {"action", "observation.images.ego"}
    ego = {"dtype": "video", "shape": [480, 640, 3], "names": ["h", "w", "c"]}
    assert feats["observation.images.ego"] == ego
    assert "fps" not in feats["action"]
