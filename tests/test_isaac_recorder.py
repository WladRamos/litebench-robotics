"""Gravador do Isaac (``scripts/isaac_recorder.py``) com um ambiente falso no estilo do Isaac Lab."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_script():
    scripts = Path(__file__).parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location("isaac_recorder", scripts / "isaac_recorder.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rec = _load_script()


class FakeIsaacEnv:
    """Estado = número da cena e step; termina com sucesso no step 3 do 1º episódio e trunca em 5.

    Como no Isaac Lab, o ``step`` que termina chama ``_reset_idx`` e devolve a cena seguinte.
    """

    def __init__(self):
        self.unwrapped = self
        self.scene_id, self.t, self.episode = 0, 0, 0
        self.scene = None

    def _reset_idx(self, env_ids=None):
        self.scene_id += 1
        self.t = 0

    def reset(self, seed=None):
        self._reset_idx()
        return {"obs": self.scene_id}, {}

    def step(self, action):
        self.t += 1
        success = self.episode == 0 and self.t == 3
        trunc = self.t == 5
        if success or trunc:
            self.episode += 1
            self._reset_idx()
        return {"subtask_terms": {"vial_grasped": np.array([[self.t >= 2]])}}, 0.0, \
            np.array([success]), np.array([trunc and not success]), {}


class FakeStage:
    def GetPrimAtPath(self, path):
        return SimpleNamespace(
            IsValid=lambda: path != "/World/sky_light",  # a Eval não tem domo
            GetAttribute=lambda name: SimpleNamespace(IsValid=lambda: True, Get=lambda: 1.5),
        )


def _state(env):
    return {"joint_pos": np.array([env.scene_id, env.t], float)}


def test_recorder_keeps_final_state_and_scene(tmp_path):
    env = FakeIsaacEnv()
    r = rec.Recorder(env, tmp_path, stage_fn=FakeStage, state_fn=_state, cameras_fn=lambda e: {})
    for _ in range(2):  # como o cliente: reset, steps até terminar, reset de novo
        r.reset()
        while True:
            _, _, term, trunc, _ = r.step(np.zeros((1, 6)))
            if term.any() or trunc.any():
                break
    lines = (tmp_path / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert [(x["success"], x["truncated"], x["steps"]) for x in rows] == [(True, False, 3), (False, True, 5)]
    for i, row in enumerate(rows):
        ep = np.load(tmp_path / f"ep_{i:04d}.npz")
        scene = row["scene"]["state0"]["joint_pos"][0]
        # o último step é o estado antes do reinício automático: a cena do episódio, no step final
        np.testing.assert_array_equal(ep["joint_pos"][:, 0], scene)
        np.testing.assert_array_equal(ep["joint_pos"][:, 1], np.arange(1, row["steps"] + 1))
        assert ep["action"].shape == (row["steps"], 6)
        # no step final, a observação é da cena seguinte: NaN
        expected = [float(t >= 2) for t in range(1, row["steps"])] + [np.nan]
        np.testing.assert_array_equal(ep["vial_grasped"], expected)
        usd = row["scene"]["usd"]
        assert usd["rect_exposure"] == 1.5 and usd["dome_exposure"] is None


class FakeCameraEnv(FakeIsaacEnv):
    """Como o ``FakeIsaacEnv``, com as duas câmeras RGB no grupo ``visual`` (o valor do pixel = step)."""

    step_dt = 1 / 30

    def _obs(self):
        img = np.full((1, 48, 64, 3), self.t, np.uint8)
        return {"visual": {"rgb_ego": img, "rgb_external": img, "depth_ego": img[..., :1]}}

    def reset(self, seed=None):
        super().reset(seed)
        return self._obs(), {}

    def step(self, action):
        o, *rest = super().step(action)
        return {**o, **self._obs()}, *rest


def test_recorder_writes_videos_of_first_episodes(tmp_path):
    av = pytest.importorskip("av")

    env = FakeCameraEnv()
    r = rec.Recorder(env, tmp_path, stage_fn=FakeStage, state_fn=_state, cameras_fn=lambda e: {}, frames=1)
    for _ in range(2):
        r.reset()
        while True:
            _, _, term, trunc, _ = r.step(np.zeros((1, 6)))
            if term.any() or trunc.any():
                break
    names = sorted(p.name for p in (tmp_path / "videos").iterdir())
    assert names == ["ep_0000_ego.mp4", "ep_0000_external.mp4"]
    with av.open(str(tmp_path / "videos" / "ep_0000_ego.mp4")) as c:
        frames = list(c.decode(video=0))
    # o reset e os 2 primeiros steps; o step que termina (3º) já mostra a cena seguinte e fica de fora
    assert len(frames) == 3 and (frames[0].width, frames[0].height) == (64, 48)
    assert json.loads((tmp_path / "meta.json").read_text())["video_fps"] == 30


def test_usd_value_and_yaw():
    q = SimpleNamespace(GetReal=lambda: np.cos(0.2), GetImaginary=lambda: (0.0, 0.0, np.sin(0.2)))
    v = rec.usd_value(q)
    assert np.isclose(rec.yaw_from_quat(v), 0.4)
    assert rec.usd_value(SimpleNamespace(path="/x/moon.exr")) == "/x/moon.exr"
    assert rec.usd_value((1, 2, 3)) == [1.0, 2.0, 3.0]


def test_vial_sdf_resolution_copies_the_usd_next_to_the_original(tmp_path):
    usd = tmp_path / "Vial_opaque.usda"
    usd.write_text('asset inputs:file = @./tex/vial_diff.png@\n'
                   '        uniform int physxSDFMeshCollision:sdfResolution = 20\n', encoding="utf-8")
    spawn = {name: SimpleNamespace(spawn=SimpleNamespace(usd_path=str(usd))) for name in rec.VIALS}
    cfg = SimpleNamespace(scene=SimpleNamespace(**spawn))
    paths = rec.set_vial_sdf_resolution(cfg, 64)
    assert paths == [str(tmp_path / "Vial_opaque.sdf64.usda")] * 3
    assert cfg.scene.vial_2.spawn.usd_path == paths[0]
    text = Path(paths[0]).read_text(encoding="utf-8")
    assert "sdfResolution = 64" in text and "@./tex/vial_diff.png@" in text  # textura relativa intacta
    assert "sdfResolution = 20" in usd.read_text(encoding="utf-8")  # o original não muda
    usd.write_text("sem o atributo\n", encoding="utf-8")
    import pytest

    with pytest.raises(ValueError, match="sdfResolution"):
        rec.usd_with_sdf_resolution(usd, 64)
