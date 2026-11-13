"""Ambiente Gymnasium Vials-Lite em MuJoCo.

observation (dict)
    "agent_pos"  float32 (6,): juntas no espaço do LeRobot (ver joint_mapping)
    "pixels"     dict com "ego" e "external_D455", uint8 (480, 640, 3); só com ``render_images=True``
    O plugin do lerobot-eval (``vials_lite.lerobot_env``) as renomeia para as chaves do dataset do
    curso (``observation.state``, ``observation.images.<câmera>``).

action
    float32 (6,) no espaço do LeRobot, posições absolutas (o dataset do curso usa esse espaço).

step
    1 step = 1/60 s = 2 passos físicos de 1/120 s (spec.TIMING) = 1 frame do dataset do curso.

info
    "is_success": bool (SuccessTracker, confirmado após 25 steps)
    "privileged": poses de frascos/rack, força garra↔frasco, flags e eventos (para o evalkit)

terminated = sucesso confirmado; truncated = 450 steps (spec.TIMING.episode_steps).

Reset (``reset_vials_rack`` e ``TaskEventCfg`` do workshop), tudo por ``self.np_random``:
rack com x ±4 cm, y ±1 cm, yaw ±0,5; com probabilidade 0,33 um frasco sorteado começa em pé num
slot sorteado; os demais com x ±4 cm, y ±1 cm, roll ±0,3 (rotação composta à direita da padrão,
como ``quat_mul(padrão, Δ)``) e z = 0,05; tapete com yaw ±0,1; focal da câmera de punho 12–15 mm;
câmera externa com ±2 cm/±2 cm/±1 cm e ±0,05 rad. Aproximação: no workshop a perturbação da câmera
externa é aplicada ao ``camera_mount``; aqui, à própria câmera. Exposição da luz da lightbox
em −3..1 (``scene.set_lighting``).

Variante ``dr-eval`` (``VialsToRackEvalDREnvCfg``): o mesmo reset, com o tapete em yaw ±0,3
e a cor do robô sorteada em ``spec.ROBOT_COLORS`` (na ``eval`` ela fica laranja, a do USD), e a
luz do domo (exposição −4..3, temperatura 2500–9500 K), que entra pela frente aberta da lightbox.
Os sorteios extras vêm depois dos da ``eval``, então a mesma semente dá a mesma cena física nas
duas. Ablação: ``dr_components`` liga só uma parte dos extras (``DR_COMPONENTS``:
domo, cor do robô, giro do tapete); os sorteios de todos continuam sendo consumidos, e os desligados
ficam como na ``eval``, então cada combinação sai pareada com a ``dr-eval`` completa (mesmos números
sorteados) e, na cena física, com a ``eval``. Luzes: ``scene.set_lighting``, com constantes
calibradas contra o vídeo do dataset (``spec.LIGHTING``).

Cena fixa: ``reset(options={"scene": {...}})`` sorteia como sempre e depois sobrescreve com a cena
dada (``SCENE_KEYS``), por exemplo a de um episódio do dataset reconstruída pelo vídeo
(``scripts/reconstruct_scenes.py``).

Condições coerentes com o dataset do curso, que foi gravado com um arranjo e uma pose inicial
diferentes dos do código do workshop. São independentes entre si e da variante:
``layout="dataset"`` (arranjo), ``init_pose="dataset"`` (pose inicial dos
operadores, ``spec.DATASET_INITIAL_ACTION``) e ``external_cam="fixed"`` (câmera externa na pose
calibrada no vídeo, sem o sorteio de ±2 cm/±0,05 rad). Os números sorteados são sempre os
mesmos: a câmera fixa consome o sorteio e o descarta, então a mesma semente dá a mesma cena com
qualquer combinação. A focal da câmera de punho continua sorteada (no dataset ela também varia:
55–64° nas cenas reconstruídas).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from . import spec
from .joint_mapping import (
    lerobot_to_radians,
    mjcf_radians_to_usd,
    radians_to_lerobot,
    usd_radians_to_mjcf,
)
from .scene import IMAGE_HEIGHT, IMAGE_WIDTH, build_model, ego_fovy_deg, physics_step, robot_rgb, set_lighting
from .success import StepState, SuccessTracker

VARIANTS = ("eval", "dr-eval")
# Extras da dr-eval que ``dr_components`` liga um a um (ablação)
DR_COMPONENTS = ("dome", "robot_color", "mat_yaw")
# Arranjo da cena em relação ao robô: o do código do workshop ou o do dataset do curso (lightbox,
# tapete e faixas do rack e dos frascos deslocados de ``spec.DATASET_LAYOUT_OFFSET``; a câmera
# externa já é a calibrada no vídeo).
LAYOUTS = ("workshop", "dataset")
# Pose inicial do braço: a do workshop (``spec.INITIAL_JOINT_POS``) ou a dos operadores do dataset
# (``spec.DATASET_INITIAL_ACTION``); câmera externa sorteada em volta da calibrada (workshop) ou fixa
# nela.
INIT_POSES = ("workshop", "dataset")
EXTERNAL_CAMS = ("random", "fixed")
GRASP_EVENT_MIN_HEIGHT = 0.055  # any_vial_grasped (observação), em vials_to_rack_env_cfg.py
GRASP_EVENT_FORCE_N = 2.0
# Chaves de ``reset(options={"scene": ...})``, todas opcionais (o que faltar fica como sorteado):
# joint_pos (6,) no espaço do LeRobot (braço e alvo dos atuadores); rack_pos, rack_quat;
# vial_pos (3, 3), vial_quat (3, 4); vials_in_rack (índices dos frascos que começam no rack);
# mat_yaw (rad, relativo ao padrão), mat_offset (dx, dy em m) e mat_half_size (meias-medidas x, y
# do tapete no mundo: o do vídeo do dataset não é o do ``mat.usda``);
# external_cam_pos, external_cam_quat, external_cam_fovy; ego_fovy (graus); robot_color; lighting
# (argumentos de ``scene.set_lighting``); lightbox_offset (dx, dy em m, absoluto: paredes, piso e
# luzes; o das cenas do dataset é ``spec.DATASET_LAYOUT_OFFSET``). ``mat_offset`` e
# ``lightbox_offset`` são relativos ao arranjo do workshop, qualquer que seja o ``layout``.
SCENE_KEYS = (
    "joint_pos", "rack_pos", "rack_quat", "vial_pos", "vial_quat", "vials_in_rack",
    "mat_yaw", "mat_offset", "mat_half_size",
    "external_cam_pos", "external_cam_quat", "external_cam_fovy", "ego_fovy", "robot_color", "lighting",
    "lightbox_offset",
)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Mesma convenção de ``isaaclab.utils.math.quat_from_euler_xyz`` (wxyz)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        ]
    )


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    u = np.array([x, y, z])
    return 2.0 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2.0 * w * np.cross(u, v)


def sample_lying_vial(rng: np.random.Generator, i: int) -> tuple[np.ndarray, np.ndarray]:
    """Pose (pos, quat wxyz) do frasco ``i`` deitado no tapete, sorteada como no reset do workshop
    (``spec.RESET.vial_pose``). Também usada pelas ablações de cena do evalkit (``--ablate vials``)."""
    v, vp = spec.VIALS[i], spec.RESET.vial_pose
    default_q = quat_from_euler_xyz(*np.deg2rad(v.euler_deg))
    delta = quat_from_euler_xyz(rng.uniform(*vp["roll"]), 0.0, rng.uniform(*vp["yaw"]))
    pos = np.array(
        [v.pos[0] + rng.uniform(*vp["x"]), v.pos[1] + rng.uniform(*vp["y"]), spec.RESET.fixed_vial_z]
    )
    return pos, _quat_mul(default_q, delta)


class GraspMonitor:
    """Porta de ``any_vial_grasped`` (mdp/terms.py): agarre com histerese, para os eventos.

    Começa a segurar quando há contato > 2 N com um frasco acima de 5,5 cm; continua enquanto houver
    contato com algum frasco; nada conta nos primeiros ``warmup_steps``.
    """

    def __init__(self, warmup_steps: int = spec.SUCCESS.warmup_steps) -> None:
        self.warmup_steps = warmup_steps
        self.reset()

    def reset(self) -> None:
        self.holding = False
        self.prev = False

    def update(self, step: int, vial_z: np.ndarray, force: np.ndarray) -> tuple[bool, list[str]]:
        if step <= 1:
            self.reset()
        contact = force > GRASP_EVENT_FORCE_N
        new_grasp = bool(np.any(contact & (vial_z > GRASP_EVENT_MIN_HEIGHT))) and not self.holding
        self.holding = (self.holding and bool(contact.any())) or new_grasp
        grasped = self.holding and step >= self.warmup_steps
        events = []
        if grasped and not self.prev:
            events.append("GRASP")
        if self.prev and not grasped:
            events.append("RELEASE")
        self.prev = grasped
        return grasped, events


class VialsLiteEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        variant: str = "eval",
        render_images: bool = True,
        render_mode: str | None = None,
        menagerie_dir: str | Path | None = None,
        history_writes_per_step: int = 2,
        actuators: str = "isaac",
        task: str | None = None,
        layout: str = "workshop",
        init_pose: str = "workshop",
        external_cam: str = "random",
        dr_components: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        """variant: "eval" (cena base) ou "dr-eval" (cor do robô, luz do domo e tapete randomizados).

        layout: "workshop" (padrão) ou "dataset" (``LAYOUTS``): lightbox, tapete, rack e frascos
        sorteados com o arranjo do dataset do curso em relação ao robô.
        init_pose: "workshop" (padrão) ou "dataset" (``INIT_POSES``): o braço parte da pose inicial
        dos operadores (``spec.DATASET_INITIAL_ACTION``); ``initial_action`` é a pose em uso.
        external_cam: "random" (padrão) ou "fixed" (``EXTERNAL_CAMS``): câmera externa sem sorteio.
        dr_components: só com ``variant="dr-eval"``: os extras ligados (``DR_COMPONENTS``); os outros
        ficam como na ``eval``, com o sorteio consumido. ``None`` (padrão) liga todos.

        actuators: "isaac" (padrão) ou "menagerie" (``scene.build_model``).
        task: texto da tarefa dado às políticas (padrão ``spec.LANGUAGE_INSTRUCTION``, a do protocolo de
        avaliação do curso).
        """
        if variant not in VARIANTS:
            raise ValueError(f"variant deve ser um de {VARIANTS}, não {variant!r}")
        if layout not in LAYOUTS:
            raise ValueError(f"layout deve ser um de {LAYOUTS}, não {layout!r}")
        if init_pose not in INIT_POSES:
            raise ValueError(f"init_pose deve ser um de {INIT_POSES}, não {init_pose!r}")
        if external_cam not in EXTERNAL_CAMS:
            raise ValueError(f"external_cam deve ser um de {EXTERNAL_CAMS}, não {external_cam!r}")
        if dr_components is not None:
            if variant != "dr-eval":
                raise ValueError("dr_components só se aplica a variant='dr-eval'")
            if set(dr_components) - set(DR_COMPONENTS):
                raise ValueError(f"dr_components: use {DR_COMPONENTS}, não {tuple(dr_components)!r}")
        import mujoco

        self._mj = mujoco
        self.variant = variant
        if variant != "dr-eval":
            self.dr_components: tuple[str, ...] = ()
        else:
            on = DR_COMPONENTS if dr_components is None else dr_components
            self.dr_components = tuple(c for c in DR_COMPONENTS if c in on)
        self.layout = layout
        offset = spec.DATASET_LAYOUT_OFFSET if layout == "dataset" else (0.0, 0.0)
        self._layout_offset = np.array([*offset, 0.0])
        self.init_pose = init_pose
        self.external_cam = external_cam
        # pose inicial no espaço do LeRobot: o lerobot_eval.py segura a initial_action no aquecimento
        if init_pose == "dataset":
            self.initial_action = np.array(spec.DATASET_INITIAL_ACTION, dtype=np.float32)
        else:
            self.initial_action = radians_to_lerobot(np.array(spec.INITIAL_JOINT_POS)).astype(np.float32)
        self.render_images = render_images
        self.render_mode = render_mode
        self.model, self.ids = build_model(
            Path(menagerie_dir) if menagerie_dir else None, actuators=actuators
        )
        self.data = mujoco.MjData(self.model)
        self.n_substeps = spec.TIMING.decimation
        self._renderer = None

        self._default_cam = {
            "ext_pos": self.model.cam_pos[self.ids.external_cam].copy(),
            "ext_quat": self.model.cam_quat[self.ids.external_cam].copy(),
            "mat_quat": self.model.geom_quat[self.ids.mat_geom].copy(),
            "mat_pos": self.model.geom_pos[self.ids.mat_geom].copy(),
            "mat_size": self.model.geom_size[self.ids.mat_geom].copy(),
            "ext_fovy": float(self.model.cam_fovy[self.ids.external_cam]),
        }
        self._lights = np.concatenate(
            [self.ids.rect_lights, [self.ids.dome_light, self.ids.fill_light]]
        )
        self._default_lightbox = {
            "geom_pos": self.model.geom_pos[self.ids.lightbox_geoms].copy(),
            "light_pos": self.model.light_pos[self._lights].copy(),
        }
        self._ctrl_lo, self._ctrl_hi = self.model.actuator_ctrlrange[self.ids.actuators].T

        lo = np.array([-100.0] * 5 + [0.0], dtype=np.float32)
        hi = np.full(6, 100.0, dtype=np.float32)
        self.action_space = gym.spaces.Box(lo, hi, dtype=np.float32)
        obs = {"agent_pos": gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)}
        if render_images:
            img = gym.spaces.Box(0, 255, (IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
            obs["pixels"] = gym.spaces.Dict({"ego": img, "external_D455": img})
        self.observation_space = gym.spaces.Dict(obs)

        self.tracker = SuccessTracker(
            n_vials=len(spec.VIALS), history_writes_per_step=history_writes_per_step
        )
        self.grasp_monitor = GraspMonitor()
        self._step = 0
        self._rack_placed_at_reset: int | list[int] | None = None  # lista: cena fixa (vials_in_rack)
        self.robot_color = spec.DEFAULT_ROBOT_COLOR
        self.mat_yaw = 0.0  # rad, relativo ao padrão
        self.lighting: dict[str, float] = {}
        # lidos pelo lerobot-eval via VectorEnv.call (instrução de texto e duração do episódio)
        self.task = self.task_description = task or spec.LANGUAGE_INSTRUCTION
        self._max_episode_steps = spec.TIMING.episode_steps

    # ------------------------------------------------------------------ reset
    def _randomize(self) -> None:
        rng = self.np_random
        m, d, ids, r = self.model, self.data, self.ids, spec.RESET

        rp = r.rack_pose
        rack_pos = np.array(spec.RACK.pos) + [rng.uniform(*rp["x"]), rng.uniform(*rp["y"]), 0.0]
        rack_pos += self._layout_offset
        rack_quat = quat_from_euler_xyz(0.0, 0.0, rng.uniform(*rp["yaw"]))  # padrão: identidade
        d.qpos[ids.rack_qpos : ids.rack_qpos + 3] = rack_pos
        d.qpos[ids.rack_qpos + 3 : ids.rack_qpos + 7] = rack_quat

        placed: int | None = None
        if rng.uniform() < r.rack_placement_prob:
            placed = int(rng.integers(len(spec.VIALS)))
            slot = np.array(spec.RACK_SLOTS_LOCAL[int(rng.integers(len(spec.RACK_SLOTS_LOCAL)))])
            a = ids.vial_qpos[placed]
            d.qpos[a : a + 3] = rack_pos + _rotate(rack_quat, slot)
            d.qpos[a + 3 : a + 7] = rack_quat  # orientação local dos slots: identidade
        self._rack_placed_at_reset = placed

        for i in range(len(spec.VIALS)):
            if i == placed:
                continue
            pos, quat = sample_lying_vial(rng, i)
            a = ids.vial_qpos[i]
            d.qpos[a : a + 3] = pos + self._layout_offset
            d.qpos[a + 3 : a + 7] = quat

        dr, on = self.variant == "dr-eval", self.dr_components
        # um só número sorteado nas duas faixas: com o giro desligado, é o da eval na mesma semente
        mat_yaw = rng.uniform(*(spec.DR.mat_yaw if "mat_yaw" in on else r.mat_yaw))
        self.mat_yaw = float(mat_yaw)
        m.geom_quat[ids.mat_geom] = _quat_mul(
            self._default_cam["mat_quat"], quat_from_euler_xyz(0, 0, mat_yaw)
        )
        m.geom_pos[ids.mat_geom] = self._default_cam["mat_pos"] + self._layout_offset
        m.geom_size[ids.mat_geom] = self._default_cam["mat_size"]
        self._place_lightbox(self._layout_offset[:2])

        m.cam_fovy[ids.ego_cam] = ego_fovy_deg(rng.uniform(*r.ego_focal_length_mm))
        m.cam_fovy[ids.external_cam] = self._default_cam["ext_fovy"]
        cp, cr = r.external_cam_pos, r.external_cam_rot
        m.cam_pos[ids.external_cam] = self._default_cam["ext_pos"] + [
            rng.uniform(*cp["x"]),
            rng.uniform(*cp["y"]),
            rng.uniform(*cp["z"]),
        ]
        delta = quat_from_euler_xyz(
            rng.uniform(*cr["roll"]), rng.uniform(*cr["pitch"]), rng.uniform(*cr["yaw"])
        )
        m.cam_quat[ids.external_cam] = _quat_mul(self._default_cam["ext_quat"], delta)
        if self.external_cam == "fixed":  # sorteio consumido e descartado: as sementes seguem alinhadas
            m.cam_pos[ids.external_cam] = self._default_cam["ext_pos"]
            m.cam_quat[ids.external_cam] = self._default_cam["ext_quat"]

        # extras da dr-eval, sorteados por último (a eval não consome esses números); na ablação, os
        # desligados são sorteados e descartados, para seguir pareados com a dr-eval completa
        self.robot_color = spec.DEFAULT_ROBOT_COLOR
        if dr and spec.DR.robot_color:
            names = list(spec.ROBOT_COLORS)
            color = names[int(rng.integers(len(names)))]
            if "robot_color" in on:
                self.robot_color = color
        m.mat_rgba[ids.robot_materials, :3] = robot_rgb(self.robot_color)
        self.lighting = {"rect_exposure": float(rng.uniform(*r.lightbox_exposure))}
        if dr:
            dome = {
                "dome_exposure": float(rng.uniform(*spec.DR.sky_exposure)),
                "temperature_k": float(rng.uniform(*spec.DR.sky_temperature_k)),
            }
            if "dome" in on:
                self.lighting.update(dome)
        set_lighting(m, ids, **self.lighting)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mj = self._mj
        mj.mj_resetData(self.model, self.data)
        if self.init_pose == "dataset":
            q0 = usd_radians_to_mjcf(lerobot_to_radians(self.initial_action.astype(np.float64)))
        else:
            q0 = usd_radians_to_mjcf(np.array(spec.INITIAL_JOINT_POS))
        self.data.qpos[self.ids.arm_qpos] = q0
        self.data.ctrl[self.ids.actuators] = np.clip(q0, self._ctrl_lo, self._ctrl_hi)
        self._randomize()
        if options and options.get("scene") is not None:
            self._apply_scene(options["scene"])
        mj.mj_forward(self.model, self.data)
        self._step = 0
        self.tracker.reset()
        self.grasp_monitor.reset()
        self._last_events: list[str] = []
        return self._observation(), self._info(success=False, force=self._jaw_forces())

    def _apply_scene(self, scene: dict) -> None:
        """Sobrescreve a cena sorteada com ``scene`` (ver ``SCENE_KEYS``)."""
        unknown = set(scene) - set(SCENE_KEYS)
        if unknown:
            raise ValueError(f"chaves desconhecidas na cena: {sorted(unknown)}")
        m, d, ids = self.model, self.data, self.ids
        if "joint_pos" in scene:
            q = usd_radians_to_mjcf(lerobot_to_radians(np.asarray(scene["joint_pos"], dtype=np.float64)))
            d.qpos[ids.arm_qpos] = q
            d.ctrl[ids.actuators] = np.clip(q, self._ctrl_lo, self._ctrl_hi)
        if "rack_pos" in scene:
            d.qpos[ids.rack_qpos : ids.rack_qpos + 3] = scene["rack_pos"]
        if "rack_quat" in scene:
            d.qpos[ids.rack_qpos + 3 : ids.rack_qpos + 7] = scene["rack_quat"]
        for key, offset, n in (("vial_pos", 0, 3), ("vial_quat", 3, 4)):
            if key in scene:
                for a, v in zip(ids.vial_qpos, np.asarray(scene[key], dtype=np.float64), strict=True):
                    d.qpos[a + offset : a + offset + n] = v
        if "vials_in_rack" in scene:
            self._rack_placed_at_reset = [int(i) for i in scene["vials_in_rack"]] or None
        if "mat_yaw" in scene:
            self.mat_yaw = float(scene["mat_yaw"])
            m.geom_quat[ids.mat_geom] = _quat_mul(
                self._default_cam["mat_quat"], quat_from_euler_xyz(0, 0, scene["mat_yaw"])
            )
        if "mat_offset" in scene:
            m.geom_pos[ids.mat_geom, :2] = self._default_cam["mat_pos"][:2] + np.asarray(scene["mat_offset"])
        if "mat_half_size" in scene:
            m.geom_size[ids.mat_geom, :2] = scene["mat_half_size"]
        if "external_cam_pos" in scene:
            m.cam_pos[ids.external_cam] = scene["external_cam_pos"]
        if "external_cam_quat" in scene:
            m.cam_quat[ids.external_cam] = scene["external_cam_quat"]
        if "external_cam_fovy" in scene:
            m.cam_fovy[ids.external_cam] = scene["external_cam_fovy"]
        if "ego_fovy" in scene:
            m.cam_fovy[ids.ego_cam] = scene["ego_fovy"]
        if "robot_color" in scene:
            self.robot_color = scene["robot_color"]
            m.mat_rgba[ids.robot_materials, :3] = robot_rgb(self.robot_color)
        if "lighting" in scene:
            self.lighting = {k: float(v) for k, v in scene["lighting"].items()}
            set_lighting(m, ids, **self.lighting)
        if "lightbox_offset" in scene:
            self._place_lightbox(scene["lightbox_offset"])

    def _place_lightbox(self, offset) -> None:
        """Lightbox (paredes, piso e luzes) deslocada de ``offset`` (dx, dy) da pose do workshop."""
        shift = np.array([*np.asarray(offset, dtype=np.float64).reshape(2), 0.0])
        m, ids = self.model, self.ids
        m.geom_pos[ids.lightbox_geoms] = self._default_lightbox["geom_pos"] + shift
        m.light_pos[self._lights] = self._default_lightbox["light_pos"] + shift

    # ------------------------------------------------------------------ step
    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(6)
        q = usd_radians_to_mjcf(lerobot_to_radians(a))
        self.data.ctrl[self.ids.actuators] = np.clip(q, self._ctrl_lo, self._ctrl_hi)
        for _ in range(self.n_substeps):
            physics_step(self.model, self.data, self.ids)
        self._step += 1

        force = self._jaw_forces()
        vial_pos, vial_quat = self._vial_poses()
        rack_pos, rack_quat = self._rack_pose()
        placed_before = self.tracker.placed_flags
        success = self.tracker.update(StepState(self._step, vial_pos, vial_quat, rack_pos, rack_quat, force))
        _, events = self.grasp_monitor.update(self._step, vial_pos[:, 2], force)
        events += [
            f"RACK:{spec.VIALS[i].name}" for i in np.flatnonzero(self.tracker.placed_flags & ~placed_before)
        ]
        self._last_events = events
        truncated = self._step >= spec.TIMING.episode_steps and not success
        return self._observation(), 0.0, bool(success), bool(truncated), self._info(success, force)

    # ------------------------------------------------------------------ leitura do estado
    def _vial_poses(self) -> tuple[np.ndarray, np.ndarray]:
        d, ids = self.data, self.ids
        return d.xpos[ids.vial_bodies].copy(), d.xquat[ids.vial_bodies].copy()

    def _rack_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[self.ids.rack_body].copy(), self.data.xquat[self.ids.rack_body].copy()

    def _jaw_forces(self) -> np.ndarray:
        """|força| de contato mandíbula móvel ↔ cada frasco (N), como o force_matrix_w do workshop."""
        m, d, ids = self.model, self.data, self.ids
        out = np.zeros(len(spec.VIALS))
        f6 = np.zeros(6)
        vial_of_body = {int(b): i for i, b in enumerate(ids.vial_bodies)}
        for k in range(d.ncon):
            c = d.contact[k]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if ids.jaw_body == b1 and b2 in vial_of_body:
                i = vial_of_body[b2]
            elif ids.jaw_body == b2 and b1 in vial_of_body:
                i = vial_of_body[b1]
            else:
                continue
            self._mj.mj_contactForce(m, d, k, f6)
            out[i] += np.linalg.norm(f6[:3])
        return out

    def joint_positions_lerobot(self) -> np.ndarray:
        q = self.data.qpos[self.ids.arm_qpos]
        return radians_to_lerobot(mjcf_radians_to_usd(q))

    def _observation(self) -> dict[str, Any]:
        obs: dict[str, Any] = {"agent_pos": self.joint_positions_lerobot().astype(np.float32)}
        if self.render_images:
            obs["pixels"] = {
                "ego": self._render_camera("ego"),
                "external_D455": self._render_camera("external_D455"),
            }
        return obs

    def _info(self, success: bool, force: np.ndarray) -> dict[str, Any]:
        vial_pos, vial_quat = self._vial_poses()
        rack_pos, rack_quat = self._rack_pose()
        return {
            "is_success": bool(success),
            "privileged": {
                "step": self._step,
                "vial_pos": vial_pos,
                "vial_quat": vial_quat,
                "rack_pos": rack_pos,
                "rack_quat": rack_quat,
                "jaw_contact_force": force,
                "gripper_pos": self.data.site_xpos[self.ids.gripper_site].copy(),
                "grasped": self.grasp_monitor.prev,
                "placed_flags": self.tracker.placed_flags,
                "confirm_counter": self.tracker.confirm_counter,
                "vial_on_rack_at_reset": self._rack_placed_at_reset,
                "robot_color": self.robot_color,
                "mat_yaw": self.mat_yaw,
                "lighting": dict(self.lighting),
                "events": list(self._last_events),
            },
        }

    # ------------------------------------------------------------------ render
    def gl_renderer(self) -> str:
        """Placa e driver que renderizam as câmeras (``GL_RENDERER (GL_VENDOR)``), melhor esforço."""
        if self._renderer is None:
            self._renderer = self._mj.Renderer(self.model, IMAGE_HEIGHT, IMAGE_WIDTH)
        try:
            from OpenGL import GL  # PyOpenGL é dependência do pacote mujoco

            self._renderer._gl_context.make_current()  # atributo interno; pode mudar entre versões
            return f"{GL.glGetString(GL.GL_RENDERER).decode()} ({GL.glGetString(GL.GL_VENDOR).decode()})"
        except Exception as e:  # noqa: BLE001 — diagnóstico, não pode derrubar o ambiente
            return f"desconhecido ({type(e).__name__})"

    def _render_camera(self, name: str) -> np.ndarray:
        if self._renderer is None:
            self._renderer = self._mj.Renderer(self.model, IMAGE_HEIGHT, IMAGE_WIDTH)
        self._renderer.update_scene(self.data, camera=name)
        return self._renderer.render().copy()

    def render(self):
        return self._render_camera("external_D455")

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
