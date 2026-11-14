"""Fumaça do runner do evalkit com uma política falsa (precisa de MuJoCo e do Menagerie)."""

import json

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")

from evalkit.runner import INITIAL_ACTION, policy_seed, rescore, run  # noqa: E402
from vials_lite import spec  # noqa: E402
from vials_lite.env import VialsLiteEnv  # noqa: E402
from vials_lite.scene import default_menagerie_dir  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (default_menagerie_dir() / "robotstudio_so101" / "so101.xml").is_file(),
    reason="MuJoCo Menagerie ausente (ver assets/README.md)",
)


class HoldPolicy:
    """Segura a pose inicial e conta quantas vezes foi consultada."""

    name = "hold"

    def __init__(self):
        self.calls = 0
        self.instructions = set()

    def reset(self):
        pass

    def act(self, observation, instruction):
        assert observation["agent_pos"].shape == (6,) and isinstance(instruction, str)
        self.calls += 1
        self.instructions.add(instruction)
        return INITIAL_ACTION


def test_runner_writes_outputs_and_skips_policy_during_warmup(tmp_path):
    assert VialsLiteEnv(render_images=False).task_description == spec.LANGUAGE_INSTRUCTION
    env = VialsLiteEnv(render_images=False, task=spec.DATASET_TASK)
    policy = HoldPolicy()
    try:
        info = run(policy, env, n_episodes=2, seed=5, out_dir=tmp_path, warmup_steps=10, max_steps=25)
    finally:
        env.close()
    assert policy.calls == 2 * 15
    assert policy.instructions == {spec.DATASET_TASK}  # a política recebe o texto do ambiente
    assert info["successes"] == 0 and info["n_episodes"] == 2
    lines = (tmp_path / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    eps = [json.loads(line) for line in lines]
    assert [e["seed"] for e in eps] == [5, 6] and all(e["steps"] == 25 for e in eps)
    assert eps[0]["failure"] == "nunca alcançou um frasco"
    arrays = np.load(tmp_path / "steps" / "ep_000.npz")
    assert arrays["vial_pos"].shape == (25, 3, 3) and arrays["phase"].shape == (25,)
    assert "**0/2**" in (tmp_path / "report.md").read_text(encoding="utf-8")
    # reclassificar dos steps gravados dá o mesmo resultado
    rescore(tmp_path)
    lines = (tmp_path / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    again = [json.loads(line) for line in lines]
    assert [e["failure"] for e in again] == [e["failure"] for e in eps]


class SeededPolicy(HoldPolicy):
    """Guarda as sementes recebidas, na ordem."""

    def __init__(self):
        super().__init__()
        self.seeds = []

    def seed(self, seed):
        self.seeds.append(seed)


def test_runner_seeds_the_policy_per_episode_and_rep(tmp_path):
    env = VialsLiteEnv(render_images=False)
    try:
        p0, p1 = SeededPolicy(), SeededPolicy()
        run(p0, env, n_episodes=2, seed=7, out_dir=tmp_path / "r0", warmup_steps=1, max_steps=2)
        info = run(p1, env, n_episodes=2, seed=7, out_dir=tmp_path / "r1", warmup_steps=1, max_steps=2, rep=1)
        run(HoldPolicy(), env, n_episodes=1, seed=7, out_dir=tmp_path / "h", warmup_steps=1, max_steps=2)
    finally:
        env.close()
    assert p0.seeds == [7, 8] and p1.seeds == [policy_seed(7, 1), policy_seed(8, 1)]
    assert len(set(p0.seeds) | set(p1.seeds)) == 4 and info["rep"] == 1
    eps = [json.loads(line) for line in (tmp_path / "r1" / "episodes.jsonl").read_text().splitlines()]
    assert [e["seed"] for e in eps] == [7, 8]  # mesmas cenas; só o sorteio da política muda
    assert [e["policy_seed"] for e in eps] == p1.seeds
    held = json.loads((tmp_path / "h" / "episodes.jsonl").read_text().splitlines()[0])
    assert held["policy_seed"] is None  # política sem seed(): nada a fixar


def test_runner_with_scenes_holds_scene_pose_in_warmup_and_records_dataset_episode(tmp_path):
    env = VialsLiteEnv(render_images=False)
    joint = (INITIAL_ACTION + np.array([0, 10.0, 0, 0, 0, 0])).tolist()
    scenes = [{"episode": 7, "scene": {"joint_pos": joint, "vials_in_rack": [0]}}]
    try:
        run(HoldPolicy(), env, n_episodes=1, seed=0, out_dir=tmp_path, warmup_steps=10, max_steps=12,
            scenes=scenes)
    finally:
        env.close()
    ep = json.loads((tmp_path / "episodes.jsonl").read_text(encoding="utf-8"))
    assert ep["dataset_episode"] == 7 and ep["vial_on_rack_at_reset"] == [0]
    actions = np.load(tmp_path / "steps" / "ep_000.npz")["action"]
    np.testing.assert_allclose(actions[:10], np.tile(joint, (10, 1)), atol=1e-5)
    np.testing.assert_allclose(actions[10:], np.tile(INITIAL_ACTION, (2, 1)), atol=1e-5)


def test_runner_holds_the_env_initial_pose_in_warmup(tmp_path):
    env = VialsLiteEnv(render_images=False, init_pose="dataset")
    try:
        run(HoldPolicy(), env, n_episodes=1, seed=0, out_dir=tmp_path, warmup_steps=10, max_steps=11)
    finally:
        env.close()
    actions = np.load(tmp_path / "steps" / "ep_000.npz")["action"]
    np.testing.assert_allclose(actions[:10], np.tile(spec.DATASET_INITIAL_ACTION, (10, 1)), atol=1e-5)


def test_ablate_scenes_swaps_only_the_requested_factors():
    from evalkit.runner import ABLATIONS, ablate_scenes
    from vials_lite import spec

    base = {"joint_pos": [0.0] * 6, "vial_pos": [[9.0, 9.0, 9.0]] * 3, "vial_quat": [[1.0, 0, 0, 0]] * 3,
            "vials_in_rack": [1], "mat_yaw": 0.1, "ego_fovy": 60.0, "lighting": {"rect_exposure": 0.2}}
    scenes = [{"episode": 3, "scene": base}]
    out = ablate_scenes(scenes, ["vials", "joint_pos", "mat"], seed=0)
    s = out[0]["scene"]
    assert out[0]["episode"] == 3 and base["vial_pos"][0] == [9.0, 9.0, 9.0]  # não altera a entrada
    np.testing.assert_allclose(s["joint_pos"], INITIAL_ACTION, atol=1e-5)
    assert s["vial_pos"][1] == [9.0, 9.0, 9.0] and s["vial_quat"][1] == [1.0, 0, 0, 0]  # no rack: fica
    for i in (0, 2):
        dx, dy = np.subtract(s["vial_pos"][i][:2], spec.VIALS[i].pos[:2])
        assert abs(dx) <= 0.04 and abs(dy) <= 0.01 and s["vial_pos"][i][2] == spec.RESET.fixed_vial_z
    assert "mat_yaw" not in s and s["ego_fovy"] == 60.0 and s["lighting"] == base["lighting"]
    assert ablate_scenes(scenes, ["vials"], seed=0)[0]["scene"]["vial_pos"] == s["vial_pos"]  # reproduzível
    assert set(ABLATIONS) == {"vials", "joint_pos", "mat", "cameras", "lighting"}
    with pytest.raises(ValueError):
        ablate_scenes(scenes, ["tapete"], seed=0)


def test_is_gpu_renderer_accepts_only_nvidia():
    from evalkit.runner import is_gpu_renderer

    assert is_gpu_renderer("NVIDIA GeForce RTX 5050 Laptop GPU/PCIe/SSE2 (NVIDIA Corporation)")
    assert not is_gpu_renderer("Mesa Intel(R) Graphics (RPL-P) (Intel)")
    assert not is_gpu_renderer("llvmpipe (LLVM 19.1.1, 256 bits) (Mesa)")
    assert not is_gpu_renderer("desconhecido (AttributeError)")


def test_isaac_replay_policy_reorders_and_converts(tmp_path):
    """Registro do Isaac: ação k no step k, juntas reordenadas pelo meta.json, radianos → LeRobot."""
    from evalkit.runner import IsaacReplayPolicy
    from vials_lite.joint_mapping import radians_to_lerobot

    names = list(spec.JOINTS_USD)[::-1]  # o Isaac grava os nomes do USD
    (tmp_path / "meta.json").write_text(json.dumps({"joint_names": names}), encoding="utf-8")
    (tmp_path / "episodes.jsonl").write_text("", encoding="utf-8")
    rad = np.linspace(-0.5, 0.5, 4 * 6).reshape(4, 6)  # ordem do Isaac (invertida)
    np.savez(tmp_path / "ep_0002.npz", action=rad)
    assert IsaacReplayPolicy.is_recording(tmp_path)
    assert not IsaacReplayPolicy.is_recording(tmp_path / "x")
    p = IsaacReplayPolicy(tmp_path)
    p.set_episode(2)
    p.reset()
    got = np.stack([p.act(None, None) for _ in range(6)])
    expected = radians_to_lerobot(rad[:, ::-1])
    np.testing.assert_allclose(got[:4], expected, rtol=1e-6)
    np.testing.assert_allclose(got[4:], expected[[-1, -1]], rtol=1e-6)  # depois do fim, a última


def test_isaac_joint_order_accepts_usd_and_lerobot_names():
    """O meta.json do Isaac traz os nomes do USD (Rotation, Pitch...); os do LeRobot também valem."""
    from evalkit.runner import isaac_joint_order

    assert isaac_joint_order(list(spec.JOINTS_USD)) == list(range(6))
    assert isaac_joint_order(list(spec.JOINTS_LEROBOT)) == list(range(6))
    # a ordem real de um registro do Isaac (meta.json) é a do USD
    names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
    assert isaac_joint_order(names[::-1]) == [5, 4, 3, 2, 1, 0]
    with pytest.raises(ValueError):
        isaac_joint_order(names[:5])
