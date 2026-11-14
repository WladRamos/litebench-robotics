"""Testes de fumaça do VialsLiteEnv. Precisam de MuJoCo, Gymnasium e do Menagerie em
assets/third_party (ou $VIALS_LITE_MENAGERIE); sem eles, são pulados (o CI roda só o núcleo)."""

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")

from vials_lite import spec  # noqa: E402
from vials_lite.env import DR_COMPONENTS, GraspMonitor, VialsLiteEnv, quat_from_euler_xyz  # noqa: E402
from vials_lite.joint_mapping import radians_to_lerobot  # noqa: E402
from vials_lite.scene import default_menagerie_dir, robot_rgb  # noqa: E402

needs_menagerie = pytest.mark.skipif(
    not (default_menagerie_dir() / "robotstudio_so101" / "so101.xml").is_file(),
    reason="MuJoCo Menagerie ausente (ver assets/README.md)",
)

INITIAL_ACTION = radians_to_lerobot(np.array(spec.INITIAL_JOINT_POS))


@pytest.fixture(scope="module")
def env():
    e = VialsLiteEnv(render_images=False)
    yield e
    e.close()


def test_quat_from_euler_matches_single_axis():
    # roll de 90°: (cos 45°, sin 45°, 0, 0)
    np.testing.assert_allclose(quat_from_euler_xyz(np.pi / 2, 0, 0), [np.sqrt(0.5), np.sqrt(0.5), 0, 0])
    np.testing.assert_allclose(quat_from_euler_xyz(0, 0, np.pi / 2), [np.sqrt(0.5), 0, 0, np.sqrt(0.5)])


def test_grasp_monitor_needs_lift_and_respects_warmup():
    g = GraspMonitor(warmup_steps=30)
    # contato sem levantar: não é agarre
    assert g.update(40, np.array([0.05, 0.05, 0.05]), np.array([5.0, 0, 0])) == (False, [])
    # contato e frasco acima de 5,5 cm: GRASP; perder o contato: RELEASE
    assert g.update(41, np.array([0.06, 0.05, 0.05]), np.array([5.0, 0, 0])) == (True, ["GRASP"])
    assert g.update(42, np.array([0.04, 0.05, 0.05]), np.array([5.0, 0, 0])) == (True, [])
    assert g.update(43, np.array([0.04, 0.05, 0.05]), np.zeros(3)) == (False, ["RELEASE"])


@needs_menagerie
def test_reset_is_deterministic_per_seed(env):
    _, a = env.reset(seed=123)
    _, b = env.reset(seed=123)
    _, c = env.reset(seed=124)
    pa, pb, pc = a["privileged"], b["privileged"], c["privileged"]
    np.testing.assert_array_equal(pa["vial_pos"], pb["vial_pos"])
    np.testing.assert_array_equal(pa["rack_quat"], pb["rack_quat"])
    assert not np.allclose(pa["rack_pos"], pc["rack_pos"])


@needs_menagerie
def test_rollout_is_deterministic(env):
    def rollout():
        obs, _ = env.reset(seed=7)
        for _ in range(40):
            obs, _, _, _, info = env.step(INITIAL_ACTION)
        return obs["agent_pos"], info["privileged"]["vial_pos"]

    (q1, v1), (q2, v2) = rollout(), rollout()
    np.testing.assert_array_equal(q1, q2)
    np.testing.assert_array_equal(v1, v2)


@needs_menagerie
def test_initial_pose_is_held_and_scene_settles(env):
    obs, info = env.reset(seed=1)  # seed 1: nenhum frasco começa no rack
    assert info["privileged"]["vial_on_rack_at_reset"] is None
    np.testing.assert_allclose(obs["agent_pos"], INITIAL_ACTION, atol=0.05)
    for _ in range(120):
        obs, _, terminated, truncated, info = env.step(INITIAL_ACTION)
    p = info["privileged"]
    assert not terminated and not truncated
    # PD do Isaac (padrão) cede à gravidade: até ~0,9 no elbow, como no dataset
    np.testing.assert_allclose(obs["agent_pos"], INITIAL_ACTION, atol=1.5)
    # frascos deitados no tapete (centro do corpo ≈ 0,035 + raio), rack apoiado no tapete
    assert np.all(np.abs(p["vial_pos"][:, 2] - 0.0504) < 0.002)
    assert abs(p["rack_pos"][2] - (0.035 - spec.RACK_BODY_Z[0])) < 0.002


@needs_menagerie
def test_truncates_at_episode_length(env):
    env.reset(seed=1)
    for k in range(spec.TIMING.episode_steps):
        _, _, terminated, truncated, _ = env.step(INITIAL_ACTION)
        assert not terminated
        assert truncated == (k == spec.TIMING.episode_steps - 1)


@needs_menagerie
def test_scripted_demo_succeeds(env):
    from vials_lite.scripted import ScriptedPickPlace

    result = ScriptedPickPlace(env).run(seed=1)
    assert result["success"], result
    assert result["steps"] < spec.TIMING.episode_steps
    names = [e for _, e in result["events"]]
    assert names[:2] == ["GRASP", "RELEASE"] and any(n.startswith("RACK:") for n in names)


@needs_menagerie
def test_isaac_actuator_preset_is_default_and_holds_initial_pose():
    from vials_lite.scene import ISAAC_EFFORT_LIMIT, ISAAC_PRESET_ARMATURE

    e = VialsLiteEnv(render_images=False)  # padrão: "isaac"
    try:
        m, ids = e.model, e.ids
        kp = [spec.ISAAC_ACTUATORS[j]["stiffness"] for j in spec.JOINTS_USD]
        np.testing.assert_allclose(m.actuator_gainprm[ids.actuators, 0], kp)
        np.testing.assert_allclose(m.actuator_forcerange[ids.actuators, 1], ISAAC_EFFORT_LIMIT)
        np.testing.assert_allclose(m.dof_armature[ids.arm_qvel], ISAAC_PRESET_ARMATURE)
        assert not m.dof_frictionloss[ids.arm_qvel].any()
        e.reset(seed=0)
        for _ in range(60):
            obs, *_ = e.step(INITIAL_ACTION)
        # PD fraco: cede à gravidade, mas pouco (o dataset mostra o mesmo)
        assert np.abs(obs["agent_pos"] - INITIAL_ACTION).max() < 3.0
    finally:
        e.close()


@needs_menagerie
def test_menagerie_actuator_preset_holds_initial_pose():
    e = VialsLiteEnv(render_images=False, actuators="menagerie")
    try:
        e.reset(seed=0)
        for _ in range(60):
            obs, *_ = e.step(INITIAL_ACTION)
        # servos STS3215 do MJCF (kp 998): praticamente rígidos
        assert np.abs(obs["agent_pos"] - INITIAL_ACTION).max() < 0.1
    finally:
        e.close()


def test_unknown_actuator_preset_rejected():
    from vials_lite.scene import build_model

    with pytest.raises(ValueError):
        build_model(actuators="foo")


def _spinning_vial_in_air(m, ids):
    """Frasco 0 no ar, longe de tudo, girando a 10 rad/s no eixo de simetria (sem giroscópio)."""
    import mujoco

    d = mujoco.MjData(m)
    q, v = ids.vial_qpos[0], ids.vial_qvel[0]
    d.qpos[q : q + 7] = [0.0, 0.0, 1.0, 1, 0, 0, 0]
    d.qvel[v + 5] = 10.0
    return d, v


@needs_menagerie
def test_physx_vial_damping_scales_rotation_before_the_step(monkeypatch):
    # PhysX: ω ← ω·(1 − c·dt) antes do solver; c = 100, dt = 1/120 → sobra 1/6 por passo
    from vials_lite import scene

    monkeypatch.setattr(scene, "VIAL_DAMPING_MODE", "physx")
    m, ids = scene.build_model()
    assert len(ids.vial_ang_dofs) == 3 * len(spec.VIALS)
    assert not m.dof_damping[ids.vial_ang_dofs].any()
    d, v = _spinning_vial_in_air(m, ids)
    scene.physics_step(m, d, ids)
    assert d.qvel[v + 5] == pytest.approx(10.0 / 6, rel=1e-6)
    scene.physics_step(m, d, ids)
    assert d.qvel[v + 5] == pytest.approx(10.0 / 36, rel=1e-6)


@needs_menagerie
def test_implicit_vial_damping_is_the_default(monkeypatch):
    # D = c·I no dof_damping com c = 100, e physics_step é só o mj_step
    from vials_lite import scene

    assert scene.VIAL_DAMPING_MODE == "implicit" and scene.VIAL_IMPLICIT_DAMPING == 100.0
    m, ids = scene.build_model()
    assert ids.vial_ang_dofs.size == 0
    v = ids.vial_qvel[0]
    np.testing.assert_allclose(
        m.dof_damping[v + 3 : v + 6], scene.VIAL_IMPLICIT_DAMPING * m.body_inertia[ids.vial_bodies[0]]
    )
    d, v = _spinning_vial_in_air(m, ids)
    scene.physics_step(m, d, ids)
    assert 10.0 / 6 < d.qvel[v + 5] < 10.0  # amortecido pelo integrador, menos que pela escala do PhysX

    monkeypatch.setattr(scene, "VIAL_DAMPING_MODE", "foo")
    with pytest.raises(ValueError):
        scene.build_model()


def test_rack_mass_isaac_is_the_mesh_volume_at_physx_default_density():
    # sem MassAPI no USD do rack, o PhysX usa 1000 kg/m³ sobre o volume da malha
    from pathlib import Path

    verts, vol = [], 0.0
    obj = Path(__file__).resolve().parents[1] / "assets" / "vials_lite" / "rack.obj"
    for line in obj.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            f = [verts[int(t.split("/")[0]) - 1] for t in line.split()[1:]]
            vol += sum(np.dot(f[0], np.cross(f[i], f[i + 1])) for i in range(1, len(f) - 1)) / 6
    assert 1000 * abs(vol) == pytest.approx(spec.RACK_MASS_ISAAC, abs=0.001)


@needs_menagerie
def test_rack_mass_mode(monkeypatch):
    from vials_lite import scene

    assert scene.RACK_MASS_MODE == "isaac"
    for mode, mass in (("declared", spec.RACK_MASS), ("isaac", spec.RACK_MASS_ISAAC)):
        monkeypatch.setattr(scene, "RACK_MASS_MODE", mode)
        m, ids = scene.build_model()
        assert m.body_mass[ids.rack_body] == pytest.approx(mass)
    monkeypatch.setattr(scene, "RACK_MASS_MODE", "foo")
    with pytest.raises(ValueError):
        scene.build_model()


@needs_menagerie
def test_vial_velocity_limit_clamps_linear_speed_only_when_enabled(monkeypatch):
    # maxLinearVelocity = 2 m/s do PhysX no frasco
    from vials_lite import scene

    assert scene.VIAL_VELOCITY_LIMIT is True
    m, ids = scene.build_model()
    for limit in (False, True):
        monkeypatch.setattr(scene, "VIAL_VELOCITY_LIMIT", limit)
        d, v = _spinning_vial_in_air(m, ids)
        d.qvel[v : v + 3] = [3.0, 4.0, 0.0]  # 5 m/s
        scene.physics_step(m, d, ids)
        speed = np.linalg.norm(d.qvel[v : v + 3])
        if limit:
            assert speed == pytest.approx(spec.VIAL_MAX_LINEAR_VELOCITY)
            assert d.qvel[v] / d.qvel[v + 1] == pytest.approx(0.75, rel=1e-3)  # mesma direção no plano
        else:
            assert speed > 4.9


def _mat_yaw(e) -> float:
    from vials_lite.env import _quat_mul

    q0 = e._default_cam["mat_quat"] * [1, -1, -1, -1]  # conjugado
    dz = _quat_mul(q0, e.model.geom_quat[e.ids.mat_geom])
    return 2 * np.arctan2(dz[3], dz[0])


@needs_menagerie
def test_dr_eval_randomizes_robot_color_and_mat_yaw():
    ev, dr = VialsLiteEnv(render_images=False), VialsLiteEnv(render_images=False, variant="dr-eval")
    try:
        colors, yaws = set(), []
        for seed in range(20):
            ev.reset(seed=seed)
            _, info = dr.reset(seed=seed)
            assert ev.robot_color == "orange"
            assert abs(_mat_yaw(ev)) <= spec.RESET.mat_yaw[1] + 1e-9
            colors.add(info["privileged"]["robot_color"])
            yaws.append(_mat_yaw(dr))
            rgba = dr.model.mat_rgba[dr.ids.robot_materials, :3]
            np.testing.assert_allclose(
                rgba, np.broadcast_to(robot_rgb(dr.robot_color), rgba.shape), rtol=1e-6
            )
            # mesma semente, mesma cena física (os sorteios da DR vêm depois)
            np.testing.assert_array_equal(ev.data.qpos, dr.data.qpos)
        assert colors <= set(spec.ROBOT_COLORS) and len(colors) >= 3
        assert max(np.abs(yaws)) <= spec.DR.mat_yaw[1] + 1e-9 and max(np.abs(yaws)) > 0.15
    finally:
        ev.close()
        dr.close()



@needs_menagerie
@pytest.mark.parametrize("components", [(), ("dome",), ("robot_color",), ("mat_yaw",), ("dome", "mat_yaw")])
def test_dr_components_switch_extras_and_stay_paired(components):
    """Ablação da dr-eval: o que está ligado é igual à dr-eval completa na mesma
    semente; o que está desligado é igual à eval; a cena física é a mesma nas três."""
    ev = VialsLiteEnv(render_images=False)
    dr = VialsLiteEnv(render_images=False, variant="dr-eval")
    ab = VialsLiteEnv(render_images=False, variant="dr-eval", dr_components=components)
    try:
        assert ab.dr_components == tuple(c for c in DR_COMPONENTS if c in components)
        for seed in range(12):
            ev.reset(seed=seed)
            _, info_dr = dr.reset(seed=seed)
            _, info = ab.reset(seed=seed)
            np.testing.assert_array_equal(ab.data.qpos, ev.data.qpos)
            p, p_dr = info["privileged"], info_dr["privileged"]
            assert p["lighting"]["rect_exposure"] == p_dr["lighting"]["rect_exposure"]
            if "dome" in components:
                assert p["lighting"] == p_dr["lighting"]
                assert ab.model.light_diffuse[ab.ids.dome_light].any()
            else:
                assert set(p["lighting"]) == {"rect_exposure"}
                assert not ab.model.light_diffuse[ab.ids.dome_light].any()
            assert p["robot_color"] == (p_dr["robot_color"] if "robot_color" in components else "orange")
            ref = dr if "mat_yaw" in components else ev
            assert p["mat_yaw"] == ref.mat_yaw
            assert abs(_mat_yaw(ab) - p["mat_yaw"]) < 1e-9
    finally:
        ev.close()
        dr.close()
        ab.close()


def test_dr_components_are_validated():
    with pytest.raises(ValueError, match="dr-eval"):
        VialsLiteEnv(render_images=False, dr_components=("dome",))
    with pytest.raises(ValueError, match="dr_components"):
        VialsLiteEnv(render_images=False, variant="dr-eval", dr_components=("hdri",))


@needs_menagerie
def test_lighting_is_randomized_within_workshop_ranges():
    ev, dr = VialsLiteEnv(render_images=False), VialsLiteEnv(render_images=False, variant="dr-eval")
    try:
        for seed in range(10):
            _, info_ev = ev.reset(seed=seed)
            _, info_dr = dr.reset(seed=seed)
            light = info_ev["privileged"]["lighting"]
            assert set(light) == {"rect_exposure"}
            lo, hi = spec.RESET.lightbox_exposure
            assert lo <= light["rect_exposure"] <= hi
            assert not ev.model.light_diffuse[ev.ids.dome_light].any()
            light = info_dr["privileged"]["lighting"]
            assert spec.DR.sky_exposure[0] <= light["dome_exposure"] <= spec.DR.sky_exposure[1]
            assert spec.DR.sky_temperature_k[0] <= light["temperature_k"] <= spec.DR.sky_temperature_k[1]
            assert dr.model.light_diffuse[dr.ids.dome_light].any()
    finally:
        ev.close()
        dr.close()


@needs_menagerie
def test_brighter_lightbox_renders_brighter_image():
    import mujoco

    from vials_lite.scene import set_lighting

    env = VialsLiteEnv(render_images=False)
    try:
        env.reset(seed=0)
        r = mujoco.Renderer(env.model, 60, 80)
        means = []
        for exposure in (-3.0, 1.0):
            set_lighting(env.model, env.ids, exposure)
            r.update_scene(env.data, camera="external_D455")
            means.append(r.render().mean())
        r.close()
        assert means[1] > means[0] + 20
    finally:
        env.close()


@needs_menagerie
def test_visual_meshes_have_no_contact_or_mass_and_collision_is_hidden():
    import mujoco

    from vials_lite.scene import HIDDEN_GROUP

    env = VialsLiteEnv(render_images=False)
    try:
        m, ids = env.model, env.ids
        bodies = [ids.rack_body, *ids.vial_bodies]
        for g in np.where(np.isin(m.geom_bodyid, bodies))[0]:
            visual = m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0
            if visual:
                assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] != HIDDEN_GROUP
            else:  # colisão: as primitivas e o cone da ponta do frasco
                assert m.geom_group[g] == HIDDEN_GROUP
        np.testing.assert_allclose(m.body_mass[ids.rack_body], spec.RACK_MASS_ISAAC)
        np.testing.assert_allclose(m.body_mass[ids.vial_bodies], spec.VIAL_MASS)
    finally:
        env.close()


def _drop_over_hole(m, d, ids, offset_mm: float) -> float:
    """Frasco 1 em pé, parado 3 mm acima do furo 0 e deslocado ``offset_mm`` para a borda; 2 s depois,
    a altura da origem dele no frame do rack."""
    from vials_lite import scene

    mujoco = pytest.importorskip("mujoco")
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    d.ctrl[ids.actuators] = d.qpos[ids.arm_qpos]
    q = ids.vial_qpos[0]
    slot = np.array(spec.RACK_SLOTS_LOCAL[0][:2])
    rack = d.qpos[ids.rack_qpos : ids.rack_qpos + 3].copy()
    z = rack[2] + spec.RACK_BODY_Z[1] - spec.VIAL_TIP_Z + 0.003
    d.qpos[q : q + 3] = [*(rack[:2] + slot + [offset_mm / 1000, 0]), z]
    d.qpos[q + 3 : q + 7] = [1, 0, 0, 0]
    d.qvel[:] = 0
    for _ in range(int(2.0 / m.opt.timestep)):
        scene.physics_step(m, d, ids)
    return float(d.qpos[q + 2] - d.qpos[ids.rack_qpos + 2])


@needs_menagerie
def test_vial_tip_cone_slides_into_the_hole_where_the_flat_cylinder_stays_on_top(monkeypatch):
    # Com a ponta em cone (USD), o padrão, o frasco solto a 24 mm do centro do furo entra pelo chanfro;
    # com a ponta em cilindro chato, fica sobre o tampo. Perto do centro, os dois entram.
    import mujoco

    from vials_lite import scene

    assert scene.VIAL_TIP_SHAPE == "cone"
    on_top = 0.08  # success_seating: origem a ≥ 0,08 m no frame do rack = sobre o tampo
    heights = {}
    for shape in scene.VIAL_TIP_SHAPES:
        monkeypatch.setattr(scene, "VIAL_TIP_SHAPE", shape)
        m, ids = scene.build_model()
        mesh = m.geom(f"{spec.VIALS[0].name}_tip").type == mujoco.mjtGeom.mjGEOM_MESH
        assert mesh == (shape == "cone")
        d = mujoco.MjData(m)
        heights[shape] = [_drop_over_hole(m, d, ids, off) for off in (5, 24)]
    assert max(heights["cone"]) < on_top
    assert heights["cylinder"][0] < on_top < heights["cylinder"][1]

    monkeypatch.setattr(scene, "VIAL_TIP_SHAPE", "foo")
    with pytest.raises(ValueError):
        scene.build_model()


def test_vial_tip_cone_top_is_trimmed_inside_the_body(monkeypatch):
    # O topo do cone vira um cilindro VIAL_TIP_CLEARANCE para dentro do corpo, para a face inclinada não
    # tocar o tapete quando o contato mole deixa o frasco afundar; a ponta é a do USD.
    import mujoco

    from vials_lite import scene

    def radii(clearance):
        monkeypatch.setattr(scene, "VIAL_TIP_CLEARANCE", clearance)
        s = mujoco.MjSpec()
        scene._add_vial_tip_mesh(s)
        v = np.reshape(s.meshes[0].uservert, (-1, 3))  # antes de compilar: o MuJoCo recentra a malha
        r = np.hypot(v[:, 0], v[:, 1])
        return r[v[:, 2] > -1e-9].max(), r[v[:, 2] < spec.VIAL_TIP_Z + 1e-9].min()

    assert scene.VIAL_TIP_CLEARANCE == 0.003
    top, tip = radii(0.003)
    np.testing.assert_allclose(top, spec.VIAL_BODY_RADIUS - 0.003, atol=1e-6)
    np.testing.assert_allclose(tip, spec.VIAL_TIP_RADIUS, atol=1e-6)
    top_full, _ = radii(None)
    np.testing.assert_allclose(top_full, spec.VIAL_BODY_RADIUS - scene.VIAL_TIP_INSET, atol=1e-6)


def _rack_top_heights(flat: bool, monkeypatch, r_mm: float = 28.0) -> list[float]:
    """Altura da colisão do rack (frame do rack) sob raios verticais a ``r_mm`` do centro do furo 0."""
    import mujoco

    from vials_lite import scene

    monkeypatch.setattr(scene, "RACK_FLAT_TOP", flat)
    m, ids = scene.build_model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    rack = d.qpos[ids.rack_qpos : ids.rack_qpos + 3].copy()
    center = rack + [*spec.RACK_SLOTS_LOCAL[0][:2], 0]
    groups = np.zeros(6, np.uint8)
    groups[scene.HIDDEN_GROUP] = 1  # só as colisões, não a malha visual
    gid = np.zeros(1, np.int32)
    heights = []
    for ang in np.radians([0, 7.5, 15, 30, 45]):
        start = center + [r_mm / 1000 * np.cos(ang), r_mm / 1000 * np.sin(ang), 0.2]
        dist = mujoco.mj_ray(m, d, start, np.array([0, 0, -1.0]), groups, 1, -1, gid)
        heights.append(float(start[2] - dist - rack[2]))
    return heights


@needs_menagerie
def test_rack_flat_top_around_the_chamfer(monkeypatch):
    # Sem o tampo plano, a 28 mm do centro do furo a colisão é a face de cima das caixas inclinadas do
    # chanfro, uma rampa ~1 mm abaixo do tampo; com ele, plana em RACK_TOP, como no USD.
    from vials_lite import scene

    assert scene.RACK_FLAT_TOP  # padrão
    flat = _rack_top_heights(True, monkeypatch)
    ramp = _rack_top_heights(False, monkeypatch)
    assert max(abs(h - scene.RACK_TOP) for h in flat) < 2e-4
    assert min(ramp) < scene.RACK_TOP - 8e-4


@needs_menagerie
def test_rack_mesh_carries_baked_ambient_occlusion():
    from vials_lite.scene import AO_FLOOR, AO_MATERIAL, AO_RAMP_TEXELS, _ao_ramp, ao_to_u

    env = VialsLiteEnv(render_images=False)
    try:
        m = env.model
        ao = m.mat(AO_MATERIAL).id
        rack = m.geom(f"{spec.RACK.name}_rack")
        assert rack.matid[0] == ao and m.mesh_texcoordnum[rack.dataid[0]] > 0
        assert m.geom(f"{spec.VIALS[0].name}_vial_body").matid[0] == -1  # só o rack tem oclusão gravada
    finally:
        env.close()
    # a rampa leva a fração do céu visível 0 → AO_FLOOR e 1 → branco, sem tocar as bordas da textura
    ramp = np.frombuffer(_ao_ramp(AO_FLOOR), np.uint8).reshape(-1, 3)[:, 0]
    texel = (ao_to_u([0.0, 1.0]) * AO_RAMP_TEXELS).astype(int)
    assert ramp[texel[0]] <= round(255 * AO_FLOOR) + 2 and ramp[texel[1]] == 255
    assert 0 < texel[0] and texel[1] < AO_RAMP_TEXELS - 1


@needs_menagerie
def test_vial_cap_carries_knurl_texture():
    from vials_lite.scene import (
        KNURL_GROOVE,
        KNURL_GROOVE_SHADE,
        KNURL_MATERIAL,
        KNURL_TEXELS,
        _knurl_texture,
    )

    env = VialsLiteEnv(render_images=False)
    try:
        m = env.model
        cap = m.geom(f"{spec.VIALS[0].name}_vial_cap")
        assert cap.matid[0] == m.mat(KNURL_MATERIAL).id
        mesh = cap.dataid[0]
        u = m.mesh_texcoord[m.mesh_texcoordadr[mesh] : m.mesh_texcoordadr[mesh] + m.mesh_texcoordnum[mesh], 0]
        assert u.max() > 99  # u em períodos de dente: 100 dentes na volta
    finally:
        env.close()
    tex = np.frombuffer(_knurl_texture(KNURL_GROOVE, KNURL_GROOVE_SHADE), np.uint8).reshape(-1, 3)[:, 0]
    dark = tex < 255
    assert len(tex) == KNURL_TEXELS and 0 < dark.mean() < 0.5
    assert tex[dark].max() == round(255 * KNURL_GROOVE_SHADE)


@needs_menagerie
def test_contacts_use_isaac_default_friction(env):
    from vials_lite.scene import GRIPPER_TORSION, ISAAC_FRICTION

    m = env.model
    assert np.all(m.geom_friction[:, 0] == ISAAC_FRICTION)
    # a garra (priority 1 no Menagerie) tem torção e não rolamento, para o frasco não pendular
    jaw = m.geom_priority == 1
    assert {m.body(b).name for b in m.geom_bodyid[jaw]} >= {"gripper", "moving_jaw_so101_v1"}
    assert np.all(m.geom_condim[jaw] == 4) and np.all(m.geom_friction[jaw, 1] == GRIPPER_TORSION)
    assert np.all(m.geom_friction[jaw, 2] == 0)


@needs_menagerie
def test_vial_placed_at_reset_seats_upright_in_the_conical_bore():
    from vials_lite import scene

    gap = 2 * (scene.RACK_BORE_RADIUS - spec.VIAL_BODY_RADIUS)
    max_tilt = np.arctan(gap / (scene.RACK_BORE_TOP - scene.RACK_FLOOR_TOP))  # ~4,3°
    env = VialsLiteEnv(render_images=False)
    try:
        seen = 0
        for seed in range(12):
            obs, info = env.reset(seed=seed)
            k = info["privileged"]["vial_on_rack_at_reset"]
            if k is None:
                continue
            for _ in range(60):
                obs, *_ = env.step(obs["agent_pos"])
            d, ids = env.data, env.ids
            rack_r = d.xmat[ids.rack_body].reshape(3, 3)
            local = rack_r.T @ (d.xpos[ids.vial_bodies[k]] - d.xpos[ids.rack_body])
            axis_z = d.xmat[ids.vial_bodies[k]].reshape(3, 3)[2, 2]
            # a ponta em cone desce no assento cônico (z ~0,015; cilindro: 0,020; piso plano: 0,030)
            assert 0.010 < local[2] < 0.020, local
            # e o corpo se inclina até encostar na parede do furo, como permite a folga (0,9–3,4°)
            assert np.arccos(min(1.0, abs(axis_z))) < max_tilt
            seen += 1
        assert seen >= 3
    finally:
        env.close()


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        VialsLiteEnv(variant="foo", render_images=False)
    with pytest.raises(ValueError):
        VialsLiteEnv(layout="foo", render_images=False)
    with pytest.raises(ValueError):
        VialsLiteEnv(init_pose="foo", render_images=False)
    with pytest.raises(ValueError):
        VialsLiteEnv(external_cam="foo", render_images=False)


@needs_menagerie
def test_dataset_init_pose_and_fixed_camera_keep_the_draw():
    """Pose dos operadores e câmera externa fixa: o resto da cena é o mesmo sorteio da semente."""
    ws = VialsLiteEnv(render_images=False)
    ds = VialsLiteEnv(render_images=False, init_pose="dataset", external_cam="fixed")
    cam = ws.model.camera("external_D455").id
    try:
        np.testing.assert_allclose(ws.initial_action, INITIAL_ACTION, atol=1e-5)
        np.testing.assert_allclose(ds.initial_action, spec.DATASET_INITIAL_ACTION, atol=1e-5)
        positions = set()
        for seed in (0, 7):
            oa, a = ws.reset(seed=seed)
            ob, b = ds.reset(seed=seed)
            np.testing.assert_allclose(oa["agent_pos"], INITIAL_ACTION, atol=0.05)
            np.testing.assert_allclose(ob["agent_pos"], spec.DATASET_INITIAL_ACTION, atol=0.05)
            pa, pb = a["privileged"], b["privileged"]
            np.testing.assert_allclose(pb["vial_pos"], pa["vial_pos"], atol=1e-9)
            np.testing.assert_allclose(pb["rack_pos"], pa["rack_pos"], atol=1e-9)
            assert pb["lighting"] == pa["lighting"]
            assert ds.model.cam_fovy[ds.ids.ego_cam] == ws.model.cam_fovy[ws.ids.ego_cam]  # punho: sorteada
            np.testing.assert_allclose(ds.model.cam_pos[cam], ds._default_cam["ext_pos"])
            np.testing.assert_allclose(ds.model.cam_quat[cam], ds._default_cam["ext_quat"])
            positions.add(tuple(ws.model.cam_pos[cam].round(6)))
        assert len(positions) == 2  # na padrão, a câmera continua sorteada
    finally:
        ws.close()
        ds.close()


@needs_menagerie
@pytest.mark.parametrize("variant", ["eval", "dr-eval"])
def test_dataset_layout_is_the_workshop_draw_shifted(variant):
    """Mesma semente: o sorteio é o mesmo, com rack, frascos, tapete e lightbox deslocados."""
    shift = np.array([*spec.DATASET_LAYOUT_OFFSET, 0.0])
    ws = VialsLiteEnv(variant=variant, render_images=False)
    ds = VialsLiteEnv(variant=variant, render_images=False, layout="dataset")
    try:
        for seed in (0, 7):
            _, a = ws.reset(seed=seed)
            _, b = ds.reset(seed=seed)
            pa, pb = a["privileged"], b["privileged"]
            np.testing.assert_allclose(pb["rack_pos"], pa["rack_pos"] + shift, atol=1e-9)
            np.testing.assert_allclose(pb["vial_pos"], pa["vial_pos"] + shift, atol=1e-9)
            assert pb["robot_color"] == pa["robot_color"] and pb["lighting"] == pa["lighting"]
            for name in ("mat", "lightbox_floor"):
                g = ws.model.geom(name).id
                np.testing.assert_allclose(ds.model.geom_pos[g], ws.model.geom_pos[g] + shift)
                np.testing.assert_allclose(ds.model.geom_quat[g], ws.model.geom_quat[g])
            for c in ("external_D455", "ego"):
                k = ws.model.camera(c).id
                np.testing.assert_allclose(ds.model.cam_pos[k], ws.model.cam_pos[k])
            # o robô não se move
            np.testing.assert_allclose(ds.data.xpos[ds.model.body("base").id],
                                       ws.data.xpos[ws.model.body("base").id])
    finally:
        ws.close()
        ds.close()


@needs_menagerie
def test_reset_with_fixed_scene_overrides_and_next_reset_restores(env):
    m, ids = env.model, env.ids
    default_fovy = float(m.cam_fovy[ids.external_cam])
    default_mat = (m.geom_pos[ids.mat_geom].copy(), m.geom_size[ids.mat_geom].copy())
    lights = [*ids.rect_lights, ids.dome_light, ids.fill_light]
    default_box = (m.geom_pos[ids.lightbox_geoms].copy(), m.light_pos[lights].copy())
    assert len(ids.lightbox_geoms) == 5
    joint = INITIAL_ACTION + np.array([5.0, -5.0, 0, 0, 0, 10.0])
    vial_pos = [[0.20, -0.15, 0.05], [0.20, -0.10, 0.05], [0.20, -0.05, 0.05]]
    scene = {
        "joint_pos": joint.tolist(),
        "rack_pos": [0.16, 0.09, 0.041],
        "vial_pos": vial_pos,
        "vials_in_rack": [1, 2],
        "mat_yaw": 0.2,
        "mat_offset": [-0.04, 0.01],
        "mat_half_size": [0.16, 0.24],
        "external_cam_fovy": 49.0,
        "robot_color": "teal",
        "lighting": {"rect_exposure": -1.0, "dome_exposure": 0.5, "temperature_k": 5000.0},
        "lightbox_offset": list(spec.DATASET_LAYOUT_OFFSET),
    }
    obs, info = env.reset(seed=3, options={"scene": scene})
    p = info["privileged"]
    np.testing.assert_allclose(obs["agent_pos"], joint, atol=1e-4)
    np.testing.assert_allclose(p["vial_pos"], vial_pos, atol=1e-9)
    np.testing.assert_allclose(p["rack_pos"], scene["rack_pos"], atol=1e-9)
    assert p["vial_on_rack_at_reset"] == [1, 2] and p["robot_color"] == "teal"
    assert p["lighting"]["dome_exposure"] == 0.5
    assert m.cam_fovy[ids.external_cam] == 49.0
    np.testing.assert_allclose(m.geom_size[ids.mat_geom, :2], [0.16, 0.24])
    np.testing.assert_allclose(m.geom_pos[ids.mat_geom, :2], default_mat[0][:2] + [-0.04, 0.01])
    shift = [*spec.DATASET_LAYOUT_OFFSET, 0.0]
    np.testing.assert_allclose(m.geom_pos[ids.lightbox_geoms], default_box[0] + shift)
    np.testing.assert_allclose(m.light_pos[lights], default_box[1] + shift)
    # o próximo reset sem cena volta ao sorteio normal, sem herdar a cena
    env.reset(seed=3)
    assert m.cam_fovy[ids.external_cam] == default_fovy
    np.testing.assert_allclose(m.geom_pos[ids.mat_geom], default_mat[0])
    np.testing.assert_allclose(m.geom_size[ids.mat_geom], default_mat[1])
    np.testing.assert_allclose(m.geom_pos[ids.lightbox_geoms], default_box[0])
    np.testing.assert_allclose(m.light_pos[lights], default_box[1])
    with pytest.raises(ValueError):
        env.reset(seed=3, options={"scene": {"vial_pose": []}})
