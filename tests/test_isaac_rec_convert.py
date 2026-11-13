"""Conversor do registro do Isaac (``scripts/isaac_rec_convert.py``), com o próprio Vials-Lite
fazendo o papel do Isaac: o estado do MuJoCo escrito no formato do gravador tem de voltar igual."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")

from vials_lite import spec  # noqa: E402
from vials_lite.env import VialsLiteEnv, _quat_mul, quat_from_euler_xyz  # noqa: E402
from vials_lite.joint_mapping import mjcf_radians_to_usd  # noqa: E402
from vials_lite.scene import default_menagerie_dir, ego_fovy_deg  # noqa: E402

needs_menagerie = pytest.mark.skipif(
    not (default_menagerie_dir() / "robotstudio_so101" / "so101.xml").is_file(),
    reason="MuJoCo Menagerie ausente (ver assets/README.md)",
)


def _load_script():
    scripts = Path(__file__).parents[1] / "scripts"
    s = importlib.util.spec_from_file_location("isaac_rec_convert", scripts / "isaac_rec_convert.py")
    mod = importlib.util.module_from_spec(s)
    sys.modules[s.name] = mod
    s.loader.exec_module(mod)
    return mod


def _isaac_state(env):
    """Estado do MuJoCo no formato do gravador (juntas em radianos do USD, ordem do LeRobot)."""
    d, ids = env.data, env.ids
    return {
        "joint_pos": mjcf_radians_to_usd(d.qpos[ids.arm_qpos].copy()),
        "vial_pos": np.stack([d.qpos[a : a + 3] for a in ids.vial_qpos]),
        "vial_quat": np.stack([d.qpos[a + 3 : a + 7] for a in ids.vial_qpos]),
        "rack_pos": d.qpos[ids.rack_qpos : ids.rack_qpos + 3].copy(),
        "rack_quat": d.qpos[ids.rack_qpos + 3 : ids.rack_qpos + 7].copy(),
    }


@needs_menagerie
def test_kinematics_and_scene_round_trip():
    conv = _load_script()
    kin = conv.Kinematics()
    order = conv.joint_order(list(conv.JOINTS))
    src = VialsLiteEnv(render_images=False, variant="dr-eval")
    dst = VialsLiteEnv(render_images=False)
    try:
        found_in_rack = False
        for seed in range(8):
            src.reset(seed=seed)
            # juntas fora da pose inicial, sem tirar do lugar o frasco do rack (metas aleatórias em todo
            # o espaço de ação arrancam o frasco em 5 steps; no Isaac a cena é lida logo depois do reset)
            rng = np.random.default_rng(seed)
            for _ in range(5):
                src.step(src.initial_action + rng.uniform(-10, 10, 6).astype(np.float32))
            s0 = _isaac_state(src)
            # cinemática direta = site do MuJoCo (depois do mj_step, o site_xpos é o do início do passo)
            src._mj.mj_kinematics(src.model, src.data)
            np.testing.assert_allclose(kin.gripper_pos(s0["joint_pos"][None])[0],
                                       src.data.site_xpos[src.ids.gripper_site], atol=1e-9)
            usd = {
                "robot_rgb": list(spec.ROBOT_COLORS[src.robot_color]),
                "mat_orient": quat_from_euler_xyz(0, 0, src.mat_yaw + np.pi / 2).tolist(),
                "ego_focal_mm": 13.0,
                "rect_exposure": src.lighting["rect_exposure"],
                "dome_exposure": src.lighting["dome_exposure"],
                "dome_temperature_k": src.lighting["temperature_k"],
                "ext_mount_translate": [0.01, -0.02, 0.005],
                "ext_mount_orient": quat_from_euler_xyz(0.0, 0.03, 0.0).tolist(),
            }
            row = {"episode": seed, "scene": {"usd": usd,
                                              "state0": {k: v.tolist() for k, v in s0.items()}}}
            mount = {"cam": {"pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]}}
            scene = conv.scene_row(row, kin, order, mount)["scene"]
            _, info = dst.reset(seed=100 + seed, options={"scene": scene})
            np.testing.assert_allclose(_isaac_state(dst)["vial_pos"], s0["vial_pos"], atol=1e-9)
            np.testing.assert_allclose(dst.data.qpos[dst.ids.arm_qpos], src.data.qpos[src.ids.arm_qpos],
                                       atol=1e-9)
            assert dst.robot_color == src.robot_color
            assert np.isclose(dst.mat_yaw, src.mat_yaw)
            assert dst.lighting == pytest.approx(src.lighting)
            assert np.isclose(dst.model.cam_fovy[dst.ids.ego_cam], ego_fovy_deg(13.0))
            np.testing.assert_allclose(dst.model.cam_pos[dst.ids.external_cam],
                                       kin.ext_cam_pos + [0.01, -0.02, 0.005])
            np.testing.assert_allclose(dst.model.cam_quat[dst.ids.external_cam],
                                       _quat_mul(kin.ext_cam_quat, quat_from_euler_xyz(0.0, 0.03, 0.0)),
                                       atol=1e-12)
            placed = src._rack_placed_at_reset
            # o frasco posto no rack pelo reset continua no rack depois de 5 steps
            assert scene["vials_in_rack"] == ([] if placed is None else [placed])
            found_in_rack |= placed is not None
        assert found_in_rack  # a amostra cobre o caso do frasco no rack
    finally:
        src.close()
        dst.close()


@needs_menagerie
def test_converted_summary_matches_runner(tmp_path):
    """O resumo do evalkit sobre o estado no formato do Isaac é o mesmo que o runner calcula."""
    from evalkit.runner import run_episode

    conv = _load_script()
    kin = conv.Kinematics()
    order = conv.joint_order(list(conv.JOINTS))
    env = VialsLiteEnv(render_images=False)

    class Wiggle:
        def reset(self):
            self.k = 0

        def act(self, obs, instruction):
            self.k += 1
            a = env.initial_action.copy()
            a[1] += 20 * np.sin(self.k / 20)
            a[5] = 50 + 40 * np.sin(self.k / 15)
            return a

    try:
        ep = run_episode(env, Wiggle(), seed=3, warmup_steps=10, max_steps=120)
        arr = ep["_arrays"]
        # o mesmo episódio no formato do gravador
        from vials_lite.joint_mapping import lerobot_to_radians

        rec = {
            "joint_pos": lerobot_to_radians(arr["agent_pos"].astype(np.float64)),
            "action": lerobot_to_radians(arr["action"].astype(np.float64)),
            "vial_pos": arr["vial_pos"], "vial_quat": arr["vial_quat"],
            "rack_pos": arr["rack_pos"], "rack_quat": arr["rack_quat"],
            "jaw_force": arr["jaw_contact_force"],
        }
        s0 = {k: rec[k][0].tolist() for k in ("joint_pos", "vial_pos", "vial_quat", "rack_pos", "rack_quat")}
        row = {"episode": 0, "success": ep["success"], "truncated": not ep["success"],
               "scene": {"usd": {}, "state0": s0}}
        summary, arrays = conv.convert_episode(row, rec, kin, order)
        np.testing.assert_allclose(arrays["gripper_pos"], arr["gripper_pos"], atol=2e-3)
        for k in ("max_phase", "failure", "n_grasps", "n_releases_outside_rack"):
            assert summary[k] == ep[k], k
    finally:
        env.close()
