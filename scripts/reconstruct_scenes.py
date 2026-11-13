"""Reconstrói a cena inicial de cada episódio do dataset do curso pelo vídeo.

O dataset grava juntas, ações e vídeo, mas não as poses dos objetos, e foi gravado na variante DR:
tapete girado, rack e frascos sorteados, às vezes um ou dois frascos já no rack, cor do robô e luz
variando. Para comparar o Vials-Lite com o Isaac no mesmo estado (teste offline da política,
``offline_gap.py``) e rodar as ações gravadas em malha aberta (``evalkit.runner --replay``), este
script estima, em duas etapas:

1. pelo 1º frame da câmera externa (640×480 reduzido a 320×240), uma cena inicial:

* câmera externa: pose e FOV de ``fit_external_camera.py`` (``--camera``, um JSON por episódio);
* rack (x, y, yaw): 1 − IoU entre o amarelo do vídeo e a silhueta do rack, Nelder-Mead;
* frascos: slots do rack ocupados (a tampa de um frasco no slot cai sobre azul); frascos deitados
  por busca em grade da tampa azul em todo o tapete, e, se faltar algum (tapado pelo braço), nos
  frames seguintes até o agarre;
* tapete (yaw): 1 − IoU entre o escuro do vídeo e o tapete, em grade, com os objetos no lugar;
* cor do robô: pela mediana RGB dos pixels do robô dividida pela do piso (``robot_color``);

2. ajuste conjunto (``MultiView``) no frame externo e em ``EGO_VIEWS`` frames da câmera de punho
   antes do agarre, cuja pose sai da cinemática: x, y e yaw de cada frasco deitado, pose do rack e
   focal da câmera de punho (12–15 mm no workshop), por 1 − IoU das silhuetas de frascos (tampa
   azul + corpo claro sobre o tapete) e do rack. A câmera externa sozinha erra a profundidade, que
   é o y do mundo; a de punho resolve isso e acha frascos que o rack ou o braço tapam.
   Refinar também a câmera externa pela silhueta do robô piorou o resultado e foi descartado.

Depois:

* lightbox deslocada de ``spec.DATASET_LAYOUT_OFFSET`` (no vídeo ela está 5 cm mais longe do
  robô que no código do workshop);
* luz (exposição da fita e do domo, temperatura): casa a cor média do piso, do fundo e do tapete,
  medidas no vídeo (mediana dos frames antes do agarre) com as máscaras da cena reconstruída
  (``appearance_target``; as de ``measure_appearance.py``, com o tapete na pose padrão, misturavam
  tapete e piso);
* assentamento: a cena roda 0,5 s com o braço parado em ``state[0]``, e as poses finais é que vão
  para a saída.

O robô no render (com ``state[0]``) tapa o que ele tapa no vídeo; a máscara do robô, dilatada, fica
fora das IoU. Conferência independente do vídeo: a ponta da garra no início do agarre (pela
cinemática de ``observation.state``, ``replay_dataset.held_interval``) deve cair no eixo de um
frasco reconstruído, e a soltura perto de um slot (``check`` na saída).

Uso (uma linha; bash e PowerShell iguais; no Ubuntu defina ``MUJOCO_GL=egl``):
    uv run --with av --with pyarrow --with scipy --with pillow python scripts/reconstruct_scenes.py
        datasets/so101_teleop_vials_rack_left --out outputs/scenes/scenes.jsonl
        [--camera outputs/camera_fit/pass2_fov51.5.jsonl]
        [--episodes 0 5] [--workers 8] [--overlays outputs/scenes/overlays]

Só a lightbox e a luz de cenas já reconstruídas (as poses ficam; minutos em vez de horas):
    uv run --with av --with pyarrow --with scipy --with pillow python scripts/reconstruct_scenes.py
        datasets/so101_teleop_vials_rack_left --relight outputs/scenes/scenes_v3.jsonl
        --out outputs/scenes/scenes_v4.jsonl --workers 8

Saída: um JSON por episódio com ``scene`` (as chaves de ``VialsLiteEnv`` ``reset(options={"scene":
...})``, ver ``env.SCENE_KEYS``), ``fit`` (qualidade de cada ajuste) e ``check``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

WIDTH, HEIGHT = 320, 240
VIDEO_KEY = "observation.images.external_D455"
# Pose padrão da câmera externa quando o pass2 do fit_external_camera.py rodou (a extraída do USD,
# antes da calibração). As linhas antigas não têm "base_pos" e o "cam_pos" delas está errado.
PASS2_BASE_POS = (0.1376, 0.3297, 0.4886)
ROBOT_MARGIN_PX = 4  # dilatação da silhueta do robô, fora das IoU
SETTLE_STEPS = 30  # 0,5 s
# busca dos frascos deitados (origem do frasco, na base do corpo; a tampa fica 0,1 m em +x)
LYING_X = np.arange(0.10, 0.3001, 0.01)
LYING_Y = np.arange(-0.25, 0.0501, 0.01)
# dois frascos deitados (eixo em x) não se sobrepõem: |Δy| ≥ ~diâmetro ou |Δx| ≥ ~comprimento. No
# dataset há frascos encostados lado a lado, não os 8 cm do spec.
LYING_MIN_SEP_Y = 0.03
LYING_MIN_SEP_X = 0.11
CAP_MIN_PIXELS = 6
EGO_KEY = "observation.images.ego"
EGO_WIDTH, EGO_HEIGHT = 160, 120
EGO_VIEWS = 6  # frames da câmera de punho entre o início e o agarre (os frascos ainda parados)
EGO_STOP_BEFORE_GRASP = 10  # a garra encosta no frasco nos últimos frames antes do bloqueio
EGO_FOCAL_MM = np.arange(12.0, 15.01, 0.25)  # o workshop sorteia 12–15 mm por episódio
REFINE_ROUNDS = 2
VIAL_YAW_MAX = 0.4  # rad; o spec diz 0, o dataset não segue o spec
ABSENT_SEARCH_GAIN = 0.02  # queda mínima da perda para aceitar um frasco achado nas vistas de punho
ABSENT_X, ABSENT_Y = -0.5, 0.6  # frasco ausente: fora da lightbox, fora das duas câmeras
LYING_MIN_SCORE = 0.3  # IoU mínima da tampa para aceitar um frasco
LATER_FRAMES_STEP = 15  # frascos tapados no 1º frame: procura a cada 15 frames até o agarre
SLOT_OCCUPIED_FRAC = 0.3  # fração da tampa simulada que cai sobre azul no vídeo
# cor do robô (robot_color): balanço de branco pelo piso; limiares nos vãos entre os grupos
TEAL_MAX_RED = 0.3  # verde-azulado: R/máx 0,04–0,17; demais ≥ 0,98
BLACK_MAX_BRIGHTNESS = 0.55  # preto: 0,23–0,47 do piso; demais ≥ 0,64
ORANGE_MIN_SATURATION = 0.4  # laranja ≥ 0,45; branco ≤ 0,37 (≥ 0,29 só sob luz quente, eps 59–62)
MAT_YAWS = np.arange(-0.45, 0.4501, 0.05)  # partidas do ajuste do tapete


def video_masks(img: np.ndarray) -> dict[str, np.ndarray]:
    """Amarelo (rack), azul (tampas) e escuro (tapete) no frame RGB uint8."""
    from PIL import Image

    hsv = np.asarray(Image.fromarray(img).convert("HSV")).astype(float)
    h, s, v = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    return {
        "yellow": (h > 25) & (h < 65) & (s > 0.7) & (v > 0.12),  # tapete sob luz quente: s ~0,55
        "blue": (h > 190) & (h < 235) & (s > 0.2) & (v > 0.25),
        "dark": v < 0.16,
    }


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def episode_camera(row: dict) -> dict:
    """Pose e FOV da câmera externa de um episódio a partir de ``dpos``/``drot_xyz`` do ajuste."""
    from vials_lite import spec
    from vials_lite.env import _quat_mul, quat_from_euler_xyz

    base_pos = np.array(row.get("base_pos", PASS2_BASE_POS))
    base_quat = np.array(row.get("base_quat_wxyz", spec.CAMERAS[1].quat_wxyz))
    return {
        "external_cam_pos": (base_pos + row["dpos"]).tolist(),
        "external_cam_quat": _quat_mul(base_quat, quat_from_euler_xyz(*row["drot_xyz"])).tolist(),
        "external_cam_fovy": float(row["fovy"]),
    }


def load_episode(data_dir: Path, ep: int) -> tuple[np.ndarray, np.ndarray]:
    """``observation.state`` e ``action`` do episódio."""
    import pyarrow.parquet as pq

    table = pq.read_table(data_dir / "data" / "chunk-000" / f"file-{ep:03d}.parquet")
    state = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float64)
    action = np.stack(table.column("action").to_numpy(zero_copy_only=False)).astype(np.float64)
    return state, action


def load_frames(data_dir: Path, ep: int, frames: list[int], key: str = VIDEO_KEY,
                size: tuple[int, int] = (WIDTH, HEIGHT)) -> dict[int, np.ndarray]:
    """Frames pedidos de uma câmera (frame i ↔ ``state[i]``), reduzidos a ``size``."""
    import av
    from PIL import Image

    video = data_dir / "videos" / key / "chunk-000" / f"file-{ep:03d}.mp4"
    want, out = set(frames), {}
    with av.open(str(video)) as c:
        for i, fr in enumerate(c.decode(c.streams.video[0])):
            if i in want:
                out[i] = np.asarray(fr.to_image().resize(size, Image.BOX))
            if i >= max(want):
                break
    return out


class SceneFitter:
    """Modelo, renderers e poses assentadas de referência (frasco deitado, frasco no slot)."""

    def __init__(self) -> None:
        import mujoco
        from measure_appearance import HEIGHT as LH
        from measure_appearance import WIDTH as LW

        from vials_lite import spec
        from vials_lite.env import VialsLiteEnv

        self.mj, self.spec = mujoco, spec
        self.env = VialsLiteEnv(render_images=False)
        self.env.reset(seed=0)
        m = self.m = self.env.model
        self.d, self.ids = self.env.data, self.env.ids
        self.seg = mujoco.Renderer(m, HEIGHT, WIDTH)
        self.seg.enable_segmentation_rendering()
        self.seg_ego = mujoco.Renderer(m, EGO_HEIGHT, EGO_WIDTH)
        self.seg_ego.enable_segmentation_rendering()
        self.rgb = mujoco.Renderer(m, LH, LW)
        base = m.body("base").id

        def in_robot(b: int) -> bool:
            while b != 0:
                if b == base:
                    return True
                b = m.body_parentid[b]
            return False

        self.robot_geoms = np.flatnonzero([in_robot(m.geom_bodyid[g]) for g in range(m.ngeom)])
        self.cap_geoms = [m.geom(f"{v.name}_vial_cap").id for v in spec.VIALS]
        self.vial_geoms = [
            (m.geom(f"{v.name}_vial_body").id, m.geom(f"{v.name}_vial_cap").id) for v in spec.VIALS
        ]
        self.rack_geom = m.geom(f"{spec.RACK.name}_rack").id
        self.ground_geoms = [m.geom("mat").id, m.geom("lightbox_floor").id]
        self.mat_quat0 = self.env._default_cam["mat_quat"].copy()
        self.mat_pos0 = self.env._default_cam["mat_pos"].copy()
        self.mat_size0 = self.env._default_cam["mat_size"].copy()
        self._settle_references()

    # ------------------------------------------------------------------ poses
    def _yaw_quat(self, yaw: float) -> np.ndarray:
        return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])

    def _settle_references(self) -> None:
        """Assenta um frasco deitado no tapete e outro no slot 0 do rack na posição padrão."""
        from vials_lite.env import _quat_mul, _rotate, quat_from_euler_xyz
        from vials_lite.scene import physics_step

        spec, d, ids = self.spec, self.d, self.ids
        rack_pos = np.array([*spec.RACK.pos[:2], 0.035 - spec.RACK_BODY_Z[0] + 0.001])
        d.qpos[ids.rack_qpos : ids.rack_qpos + 7] = [*rack_pos, 1, 0, 0, 0]
        v0, v1 = ids.vial_qpos[0], ids.vial_qpos[1]
        d.qpos[v0 : v0 + 3] = rack_pos + np.array(spec.RACK_SLOTS_LOCAL[0])
        d.qpos[v0 + 3 : v0 + 7] = [1, 0, 0, 0]
        d.qpos[v1 : v1 + 3] = [*spec.VIALS[1].pos[:2], spec.RESET.fixed_vial_z]
        d.qpos[v1 + 3 : v1 + 7] = quat_from_euler_xyz(*np.deg2rad(spec.VIALS[1].euler_deg))
        self._park(2)
        d.qvel[:] = 0
        self.mj.mj_forward(self.m, d)
        for _ in range(SETTLE_STEPS * 4 * self.env.n_substeps):
            physics_step(self.m, d, self.ids)
        rack_pos = d.qpos[ids.rack_qpos : ids.rack_qpos + 3].copy()
        rack_quat = d.qpos[ids.rack_qpos + 3 : ids.rack_qpos + 7].copy()
        inv = rack_quat * [1, -1, -1, -1]
        slot0 = np.array([*spec.RACK_SLOTS_LOCAL[0][:2], 0.0])
        self.slot_local_pos = _rotate(inv, d.qpos[v0 : v0 + 3] - rack_pos) - slot0
        self.slot_local_quat = _quat_mul(inv, d.qpos[v0 + 3 : v0 + 7].copy())
        self.rack_z = float(rack_pos[2])
        self.lying_z = float(d.qpos[v1 + 2])
        self.lying_quat = d.qpos[v1 + 3 : v1 + 7].copy()
        from vials_lite.scene import GROUND_Z

        mat_top = self.mat_pos0[2] + self.mat_size0[2]
        self.absent_z = GROUND_Z + self.lying_z - mat_top  # mesma altura sobre o chão que sobre o tapete

    def _park(self, i: int) -> None:
        a = self.ids.vial_qpos[i]
        self.d.qpos[a : a + 7] = [2.0 + 0.3 * i, 2.0, 0.1, 1, 0, 0, 0]

    def rack_pose(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(dx, dy, yaw) → posição e quat do rack."""
        pos = np.array([self.spec.RACK.pos[0] + x[0], self.spec.RACK.pos[1] + x[1], self.rack_z])
        return pos, self._yaw_quat(x[2])

    def slot_pose(self, rack_x: np.ndarray, s: int) -> tuple[np.ndarray, np.ndarray]:
        from vials_lite.env import _quat_mul, _rotate

        pos, quat = self.rack_pose(rack_x)
        local = np.array([*self.spec.RACK_SLOTS_LOCAL[s][:2], 0.0]) + self.slot_local_pos
        return pos + _rotate(quat, local), _quat_mul(quat, self.slot_local_quat)

    def place(self, q_arm, rack_x=None, vials=None, mat=(0.0, 0.0, 0.0, 1.0, 1.0)) -> None:
        """Braço, rack (None: fora), frascos (lista de (pos, quat) ou None: fora) e tapete
        (dx, dy, yaw, escala x, escala y)."""
        from vials_lite.env import _quat_mul

        d, ids = self.d, self.ids
        d.qpos[ids.arm_qpos] = q_arm
        if rack_x is None:
            d.qpos[ids.rack_qpos : ids.rack_qpos + 7] = [3.0, 3.0, 0.1, 1, 0, 0, 0]
        else:
            pos, quat = self.rack_pose(rack_x)
            d.qpos[ids.rack_qpos : ids.rack_qpos + 7] = [*pos, *quat]
        for i in range(len(self.spec.VIALS)):
            pose = None if vials is None else vials[i]
            if pose is None:
                self._park(i)
            else:
                a = ids.vial_qpos[i]
                d.qpos[a : a + 3], d.qpos[a + 3 : a + 7] = pose
        m = self.m
        m.geom_quat[ids.mat_geom] = _quat_mul(self.mat_quat0, self._yaw_quat(mat[2]))
        m.geom_pos[ids.mat_geom, :2] = self.mat_pos0[:2] + np.asarray(mat[:2])
        m.geom_size[ids.mat_geom, :2] = self.mat_size0[:2] * np.asarray(mat[3:5])
        self.mj.mj_kinematics(self.m, d)
        self.mj.mj_camlight(self.m, d)

    def labels(self, ego: bool = False) -> np.ndarray:
        """Geom visível em cada pixel da câmera externa ou da de punho (−1: nenhum)."""
        r = self.seg_ego if ego else self.seg
        r.update_scene(self.d, camera=self.ids.ego_cam if ego else self.ids.external_cam)
        s = r.render()
        return np.where(s[..., 1] == int(self.mj.mjtObj.mjOBJ_GEOM), s[..., 0], -1)

    def set_camera(self, cam: dict) -> None:
        m, c = self.m, self.ids.external_cam
        m.cam_pos[c], m.cam_quat[c], m.cam_fovy[c] = (
            cam["external_cam_pos"], cam["external_cam_quat"], cam["external_cam_fovy"],
        )


# ---------------------------------------------------------------------- ajustes
def fit_rack(f: SceneFitter, q_arm, yellow, keep) -> tuple[np.ndarray, float]:
    from scipy.optimize import minimize

    def loss(x):
        f.place(q_arm, rack_x=x)
        return 1.0 - iou(yellow & keep, (f.labels() == f.rack_geom) & keep)

    best = None
    for yaw in (-0.5, -0.25, 0.0, 0.25, 0.5):
        simplex = np.array([[0, 0, yaw], [0.02, 0, yaw], [0, 0.02, yaw], [0, 0, yaw + 0.15]])
        r = minimize(loss, simplex[0], method="Nelder-Mead",
                     options={"initial_simplex": simplex, "xatol": 5e-4, "fatol": 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    x = best.x.copy()
    x[2] = (x[2] + np.pi / 4) % (np.pi / 2) - np.pi / 4  # o rack é simétrico a 90°
    return x, 1.0 - float(best.fun)


def cap_score(f: SceneFitter, i: int, blue, keep) -> float:
    """IoU da tampa do frasco i com o azul em volta dela, só nos pixels ``keep`` (fora do robô)."""
    from scipy.ndimage import binary_dilation

    cap = (f.labels() == f.cap_geoms[i]) & keep
    if cap.sum() < CAP_MIN_PIXELS:
        return 0.0
    return iou(cap, blue & binary_dilation(cap, iterations=6))


def separated(dx, dy):
    """Dois frascos deitados com origens a (dx, dy) não se sobrepõem."""
    return (np.abs(dy) >= LYING_MIN_SEP_Y) | (np.abs(dx) >= LYING_MIN_SEP_X)


def lying_pose(f: SceneFitter, x: float, y: float, yaw: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    from vials_lite.env import _quat_mul

    return np.array([x, y, f.lying_z]), _quat_mul(f._yaw_quat(yaw), f.lying_quat)


def find_lying(f: SceneFitter, q_arm, rack_x, blue, keep, n: int, found: list) -> list[tuple]:
    """Até ``n`` frascos deitados novos (x, y, nota), longe dos ``found``: grade de 1 cm da tampa
    contra o azul, refinada em 2 mm em volta de cada pico."""
    def score(x, y):
        f.place(q_arm, rack_x=rack_x, vials=[lying_pose(f, x, y), None, None])
        return cap_score(f, 0, blue, keep)

    grid = np.array([[score(x, y) for y in LYING_Y] for x in LYING_X])
    new: list[tuple[float, float, float]] = []
    taken = [np.array(v[:2]) for v in found]
    while len(new) < n:
        a, b = np.unravel_index(np.argmax(grid), grid.shape)
        if grid[a, b] < LYING_MIN_SCORE:
            break
        x0, y0 = LYING_X[a], LYING_Y[b]
        gx, gy = np.meshgrid(LYING_X - x0, LYING_Y - y0, indexing="ij")
        grid[~separated(gx, gy)] = -1
        sc, x, y = max((score(x0 + dx, y0 + dy), x0 + dx, y0 + dy)
                       for dx in np.arange(-0.008, 0.0081, 0.002) for dy in np.arange(-0.008, 0.0081, 0.002))
        if all(separated(x - t[0], y - t[1]) for t in taken):
            new.append((float(x), float(y), float(sc)))
            taken.append(np.array([x, y]))
    return new


def slot_occupancy(f: SceneFitter, q_arm, rack_x, blue) -> list[float]:
    """Fração da tampa simulada (frasco no slot s) que cai sobre azul, por slot."""
    out = []
    for s in range(len(f.spec.RACK_SLOTS_LOCAL)):
        vials = [f.slot_pose(rack_x, s), None, None]
        f.place(q_arm, rack_x=rack_x, vials=vials)
        cap = f.labels() == f.cap_geoms[0]
        out.append(float((cap & blue).sum() / cap.sum()) if cap.sum() >= 8 else 0.0)
    return out


def fit_mat(f: SceneFitter, q_arm, rack_x, vials, dark, robot) -> tuple[np.ndarray, float]:
    """(dx, dy, yaw, escala x, escala y) do tapete: 1 − IoU com o escuro do vídeo, onde o render
    mostra tapete ou piso (fora do robô, do rack e dos frascos). Grade no yaw, depois Nelder-Mead."""
    from scipy.optimize import minimize

    def loss(x):
        f.place(q_arm, rack_x=rack_x, vials=vials, mat=x)
        lab = f.labels()
        valid = np.isin(lab, f.ground_geoms) & ~robot
        return 1.0 - iou(dark & valid, (lab == f.ground_geoms[0]) & valid)

    starts = sorted((loss([0, 0, yaw, 1, 1]), yaw) for yaw in MAT_YAWS)[:2]
    best = None
    for _, yaw in starts:
        x0 = np.array([0.0, 0.0, yaw, 1.0, 1.0])
        simplex = np.vstack([x0, x0 + np.diag([0.03, 0.03, 0.08, 0.1, 0.1])])
        r = minimize(loss, x0, method="Nelder-Mead",
                     options={"initial_simplex": simplex, "xatol": 1e-3, "fatol": 1e-4, "maxfev": 600})
        if best is None or r.fun < best.fun:
            best = r
    return best.x, 1.0 - float(best.fun)


def robot_color(img, robot_exact, floor) -> tuple[str, list[float]]:
    """Cor da paleta pela mediana RGB dos pixels do robô (silhueta erodida) dividida pela do piso,
    que é branco: o balanço de branco tira o tom da luz do episódio.

    Verde-azulado: vermelho < ``TEAL_MAX_RED`` do máximo; preto: brilho relativo ao piso <
    ``BLACK_MAX_BRIGHTNESS``; laranja: saturação > ``ORANGE_MIN_SATURATION``; senão, branco. Os
    limiares estão nos vãos entre os grupos dos 75 episódios. Comparar com o render do robô em cada
    cor não serve: no vídeo o robô sai mais claro em relação ao piso que no MuJoCo (até 1,2 × o piso,
    contra 0,8 no render) e quase tudo viraria branco.
    """
    from scipy.ndimage import binary_erosion

    rgb = np.median(img[binary_erosion(robot_exact, iterations=3)], axis=0).astype(float)
    wb = rgb / np.maximum(np.median(img[floor], axis=0).astype(float), 1.0)
    brightness = float(wb.max())
    saturation = 1.0 - float(wb.min()) / max(brightness, 1e-6)
    if wb[0] < TEAL_MAX_RED * brightness:
        color = "teal"
    elif brightness < BLACK_MAX_BRIGHTNESS:
        color = "black"
    else:
        color = "orange" if saturation > ORANGE_MIN_SATURATION else "white"
    return color, [*np.round(wb, 3).tolist(), round(saturation, 3)]


def appearance_target(f: SceneFitter, frames: list[np.ndarray]) -> tuple[dict, dict]:
    """Máscaras de piso, fundo e tapete da cena reconstruída (160×120, longe dos objetos) e a cor
    média aparada de cada uma no vídeo, na mediana dos ``frames`` (o braço se move, o fundo não)."""
    from measure_appearance import region_masks, region_means

    masks = region_masks(f.m, f.d, "external_D455", 3)
    h, w = next(iter(masks.values())).shape
    small = [im.reshape(h, im.shape[0] // h, w, im.shape[1] // w, 3).mean((1, 3)) for im in frames]
    return masks, region_means({"default": np.median(np.stack(small), axis=0)}, masks)


def fit_lighting(f: SceneFitter, masks: dict, target: dict) -> tuple[dict, float]:
    """Exposição da fita e do domo e temperatura que casam piso, fundo e tapete com o vídeo."""
    from measure_appearance import region_means
    from scipy.optimize import minimize

    from vials_lite.scene import set_lighting

    keys = [k for k in ("floor", "back", "mat") if k in target]
    ref = np.array([target[k] for k in keys])
    lo, hi = np.array([-3.0, -4.0, 2500.0]), np.array([1.0, 3.0, 9500.0])

    def unpack(z):
        x = lo + (hi - lo) * np.clip(z, 0, 1)
        return {"rect_exposure": float(x[0]), "dome_exposure": float(x[1]), "temperature_k": float(x[2])}

    def loss(z):
        set_lighting(f.m, f.ids, **unpack(z))
        f.rgb.update_scene(f.d, camera="external_D455")
        sim = region_means({"default": f.rgb.render().astype(float)}, {k: masks[k] for k in keys})
        return float(np.mean(np.abs(np.array([sim[k] for k in keys]) - ref)))

    best = None
    for z0 in ([0.5, 0.5, 0.5], [0.8, 0.2, 0.5], [0.2, 0.8, 0.5]):
        r = minimize(loss, z0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 0.05, "maxfev": 200})
        if best is None or r.fun < best.fun:
            best = r
    return unpack(best.x), float(best.fun)


def check(f: SceneFitter, scene: dict, state: np.ndarray, action: np.ndarray) -> dict:
    """Agarre e soltura pela cinemática do estado gravado contra a cena reconstruída."""
    from replay_dataset import held_interval

    from evalkit.progress import distance_to_vial_axis
    from vials_lite.env import _rotate
    from vials_lite.joint_mapping import lerobot_to_radians, usd_radians_to_mjcf

    iv = held_interval(state[:, 5], action[:, 5])
    if iv is None:
        return {"held": None}

    def tip(t):
        f.d.qpos[f.ids.arm_qpos] = usd_radians_to_mjcf(lerobot_to_radians(state[t]))
        f.mj.mj_kinematics(f.m, f.d)
        return f.d.site_xpos[f.ids.gripper_site].copy()

    g = tip(iv[0])
    dists = [distance_to_vial_axis(g, np.array(p), np.array(q))
             for p, q in zip(scene["vial_pos"], scene["vial_quat"], strict=True)]
    k = int(np.argmin(dists))
    r = tip(min(iv[1], len(state) - 1))
    inv = np.array(scene["rack_quat"]) * [1, -1, -1, -1]
    local = _rotate(inv, r - np.array(scene["rack_pos"]))
    slots = np.array(f.spec.RACK_SLOTS_LOCAL)[:, :2]
    return {
        "held": [int(iv[0]), int(iv[1])],
        "grasp_tip": g.round(4).tolist(),
        "grasped_vial": k,
        "grasp_axis_dist_m": round(float(dists[k]), 4),
        "grasped_vial_in_rack": k in scene["vials_in_rack"],
        "release_slot_dist_m": round(float(np.min(np.linalg.norm(slots - local[:2], axis=1))), 4),
    }


# ---------------------------------------------------------------------- ajuste em várias vistas
@dataclass
class View:
    """Um frame (externo ou de punho) com o braço em ``q_arm`` e as máscaras do vídeo."""

    ego: bool
    q_arm: np.ndarray
    vial: np.ndarray  # tampas azuis + corpo claro sobre o tapete
    rack: np.ndarray  # amarelo
    keep: np.ndarray  # fora da silhueta (dilatada) do robô
    weight: float


def make_view(f: SceneFitter, img: np.ndarray, q_arm, ego: bool, weight: float, objects: dict) -> View:
    """Máscaras do vídeo; a região "sobre o tapete" e a do robô vêm do render da cena inicial."""
    from PIL import Image
    from scipy.ndimage import binary_dilation

    f.place(q_arm, **objects)
    lab = f.labels(ego)
    robot = binary_dilation(np.isin(lab, f.robot_geoms), iterations=2 if ego else ROBOT_MARGIN_PX)
    vial_geoms = [g for v in f.vial_geoms for g in v]
    on_mat = binary_dilation(np.isin(lab, [f.ground_geoms[0], f.rack_geom, *vial_geoms]), iterations=4)
    hsv = np.asarray(Image.fromarray(img).convert("HSV")).astype(float) / 255
    sat, val = hsv[..., 1], hsv[..., 2]
    vm = video_masks(img)
    mat_px = (lab == f.ground_geoms[0]) & vm["dark"]
    mat_v = float(np.median(val[mat_px])) if mat_px.sum() > 20 else 0.1
    bright = (sat < 0.3) & (val > mat_v + 0.15)  # corpo do frasco: claro e pouco saturado, sobre o tapete
    vial = vm["blue"] | (bright & on_mat & ~vm["yellow"])
    return View(ego, q_arm, vial, vm["yellow"], ~robot, weight)


def multiview_loss(f: SceneFitter, views: list[View], objects: dict) -> float:
    """Soma ponderada de (1 − IoU) dos frascos e do rack em cada vista."""
    vial_geoms = [g for v in f.vial_geoms for g in v]
    total = 0.0
    for v in views:
        f.place(v.q_arm, **objects)
        lab = f.labels(v.ego)
        keep = v.keep
        sim_vial = np.isin(lab, vial_geoms) & keep
        sim_rack = (lab == f.rack_geom) & keep
        loss = 1.0 - iou(v.vial & keep, sim_vial)
        if sim_rack.any() or (v.rack & keep).any():
            loss += 1.0 - iou(v.rack & keep, sim_rack)
        total += v.weight * loss
    return total


class MultiView:
    """Frascos deitados (x, y, yaw), rack (dx, dy, yaw) e focal da câmera de punho, ajustados juntos
    no frame externo e em ``EGO_VIEWS`` frames de punho antes do agarre (os objetos ainda parados)."""

    def __init__(self, f: SceneFitter, lying: np.ndarray, occupied: list[int], rack_x: np.ndarray,
                 mat: np.ndarray, n_absent: int):
        from vials_lite.scene import ego_fovy_deg

        self.f, self.lying, self.occupied = f, lying.copy(), occupied
        self.rack_x, self.mat, self.n_absent = rack_x.copy(), mat, n_absent
        self.focal = 13.5
        self._fovy = ego_fovy_deg
        self.views: list[View] = []

    def objects(self, lying=None, rack_x=None) -> dict:
        f = self.f
        lying = self.lying if lying is None else lying
        rack_x = self.rack_x if rack_x is None else rack_x
        vials = [lying_pose(f, *v) for v in lying] + [f.slot_pose(rack_x, k) for k in self.occupied]
        for j in range(len(f.spec.VIALS) - len(vials)):
            vials.append((np.array([ABSENT_X, ABSENT_Y + 0.15 * j, f.absent_z]), f.lying_quat.copy()))
        return {"rack_x": rack_x, "vials": vials, "mat": self.mat}

    def loss(self, lying=None, rack_x=None, focal=None) -> float:
        lying = self.lying if lying is None else lying
        if len(lying) and np.max(np.abs(lying[:, 2])) > VIAL_YAW_MAX:
            return 10.0
        for a in range(len(lying)):
            for b in range(a + 1, len(lying)):
                if not separated(lying[a, 0] - lying[b, 0], lying[a, 1] - lying[b, 1]):
                    return 10.0
        self.f.m.cam_fovy[self.f.ids.ego_cam] = self._fovy(self.focal if focal is None else focal)
        return multiview_loss(self.f, self.views, self.objects(lying, rack_x))

    def _nm(self, fun, x0, steps, maxfev=120):
        from scipy.optimize import minimize

        simplex = np.vstack([x0, x0 + np.diag(steps)])
        r = minimize(fun, x0, method="Nelder-Mead",
                     options={"initial_simplex": simplex, "xatol": 5e-4, "fatol": 1e-4, "maxfev": maxfev})
        return r.x, float(r.fun)

    def refine_round(self) -> None:
        for i in range(len(self.lying)):
            def fun(z, i=i):
                lying = self.lying.copy()
                lying[i] = z
                return self.loss(lying)

            self.lying[i], _ = self._nm(fun, self.lying[i].copy(), [0.01, 0.01, 0.15])
        self.rack_x, _ = self._nm(lambda z: self.loss(rack_x=z), self.rack_x.copy(), [0.01, 0.01, 0.1])
        self.focal = float(min(EGO_FOCAL_MM, key=lambda fm: self.loss(focal=fm)))

    def search_absent(self) -> int:
        """Procura, em todas as vistas, frascos que o 1º frame externo não mostrou."""
        found = 0
        for _ in range(self.n_absent):
            base = self.loss()
            best = (base, None)
            for x in LYING_X:
                for y in LYING_Y:
                    cand = np.vstack([self.lying, [x, y, 0.0]])  # lying é sempre (n, 3)
                    best = min(best, (self.loss(cand), (x, y)), key=lambda b: b[0])
            if best[1] is None or base - best[0] < ABSENT_SEARCH_GAIN:
                break
            self.lying = np.vstack([self.lying, [*best[1], 0.0]])
            found += 1
        self.n_absent -= found
        return found


# ---------------------------------------------------------------------- episódio
_fitter: SceneFitter | None = None


def reconstruct(job: dict) -> dict:
    from replay_dataset import held_interval
    from scipy.ndimage import binary_dilation

    from vials_lite.joint_mapping import lerobot_to_radians, usd_radians_to_mjcf

    global _fitter
    if _fitter is None:
        _fitter = SceneFitter()
    f = _fitter
    t0 = time.time()
    ep, data = job["episode"], Path(job["data"])
    state, action = load_episode(data, ep)
    held = held_interval(state[:, 5], action[:, 5])
    later = list(range(LATER_FRAMES_STEP, (held[0] if held else len(state)) - 5, LATER_FRAMES_STEP))
    frames = load_frames(data, ep, [0, *later])
    cam = episode_camera(job["camera"])
    f.set_camera(cam)

    def arm(t):
        return usd_radians_to_mjcf(lerobot_to_radians(state[t]))

    def robot_mask(t):
        f.place(arm(t))
        exact = np.isin(f.labels(), f.robot_geoms)
        return exact, binary_dilation(exact, iterations=ROBOT_MARGIN_PX)

    img, q_arm = frames[0], arm(0)
    vm = video_masks(img)
    robot_exact, robot = robot_mask(0)
    rack_x, rack_iou = fit_rack(f, q_arm, vm["yellow"], ~robot)
    occupancy = slot_occupancy(f, q_arm, rack_x, vm["blue"] & ~robot)
    occupied = sorted(np.flatnonzero(np.array(occupancy) > SLOT_OCCUPIED_FRAC).tolist(),
                      key=lambda k: -occupancy[k])[:2]
    # o azul das tampas no rack não é de frasco deitado
    rack_caps = np.zeros_like(robot)
    for k in occupied:
        f.place(q_arm, rack_x=rack_x, vials=[f.slot_pose(rack_x, k), None, None])
        rack_caps |= binary_dilation(f.labels() == f.cap_geoms[0], iterations=4)

    n_lying = len(f.spec.VIALS) - len(occupied)
    lying = find_lying(f, q_arm, rack_x, vm["blue"] & ~rack_caps, ~robot, n_lying, [])
    seen_at = [0] * len(lying)
    for t in later:  # tapados pelo braço no 1º frame: procura enquanto ele se move, antes do agarre
        if len(lying) >= n_lying:
            break
        _, robot_t = robot_mask(t)
        blue_t = video_masks(frames[t])["blue"] & ~rack_caps
        new = find_lying(f, arm(t), rack_x, blue_t, ~robot_t, n_lying - len(lying), lying)
        lying += new
        seen_at += [t] * len(new)
    missing = n_lying - len(lying)
    order = np.argsort([y for _, y, _ in lying])
    lying, seen_at = [lying[k] for k in order], [seen_at[k] for k in order]
    vials = [lying_pose(f, x, y) for x, y, _ in lying] + [f.slot_pose(rack_x, k) for k in occupied]
    in_rack = list(range(len(lying), len(vials)))
    # não visto em frame nenhum antes do agarre (nem com o braço fora da frente): fica fora da cena,
    # deitado no chão atrás da câmera, em vez de inventado num lugar padrão
    for j in range(missing):
        vials.append((np.array([ABSENT_X, ABSENT_Y + 0.15 * j, f.absent_z]), f.lying_quat.copy()))
    mat_x, mat_iou = fit_mat(f, q_arm, rack_x, vials, vm["dark"], robot)

    # ajuste conjunto no frame externo e em frames de punho antes do agarre (o externo erra
    # a profundidade, que é o y do mundo); frascos tapados no externo podem aparecer no de punho
    mv = MultiView(f, np.array([[x, y, 0.0] for x, y, _ in lying]).reshape(-1, 3), occupied, rack_x,
                   mat_x, missing)
    end = (held[0] if held else len(state)) - EGO_STOP_BEFORE_GRASP
    ego_t = sorted({int(t) for t in np.linspace(0, max(end, 1), EGO_VIEWS).round()})
    ego_frames = load_frames(data, ep, ego_t, EGO_KEY, (EGO_WIDTH, EGO_HEIGHT))
    f.m.cam_fovy[f.ids.ego_cam] = mv._fovy(mv.focal)
    start = mv.objects()
    mv.views = [make_view(f, img, q_arm, False, 1.0, start)] + [
        make_view(f, ego_frames[t], arm(t), True, 1.0 / len(ego_t), start) for t in ego_t
    ]
    loss_before = mv.loss()
    found_ego = 0
    for k in range(REFINE_ROUNDS):
        mv.refine_round()
        if k == 0 and mv.n_absent:
            found_ego = mv.search_absent()
    loss_after = mv.loss()
    seen_at += ["punho"] * found_ego
    lying_xyyaw, rack_x, missing = mv.lying, mv.rack_x, mv.n_absent
    vials = mv.objects()["vials"]
    in_rack = list(range(len(lying_xyyaw), len(lying_xyyaw) + len(occupied)))

    # assenta com o braço parado em state[0] e lê as poses finais
    rack_pos, rack_quat = f.rack_pose(rack_x)
    scene = {
        "joint_pos": state[0].tolist(),
        "rack_pos": rack_pos.tolist(),
        "rack_quat": rack_quat.tolist(),
        "vial_pos": [p.tolist() for p, _ in vials],
        "vial_quat": [q.tolist() for _, q in vials],
        "vials_in_rack": in_rack,
        "mat_yaw": float(mat_x[2]),
        "mat_offset": mat_x[:2].round(4).tolist(),
        "mat_half_size": (f.mat_size0[:2] * mat_x[3:5]).round(4).tolist(),
        **cam,
        "ego_fovy": float(mv._fovy(mv.focal)),
        "lightbox_offset": list(f.spec.DATASET_LAYOUT_OFFSET),
    }
    env = f.env
    env.reset(seed=0, options={"scene": scene})
    for _ in range(SETTLE_STEPS):
        env.step(np.asarray(state[0], dtype=np.float32))
    vial_pos, vial_quat = env._vial_poses()
    rack_pos, rack_quat = env._rack_pose()
    settle_shift = float(np.max(np.linalg.norm(vial_pos - np.array([p for p, _ in vials]), axis=1)))
    scene.update(vial_pos=vial_pos.round(5).tolist(), vial_quat=vial_quat.round(5).tolist(),
                 rack_pos=rack_pos.round(5).tolist(), rack_quat=rack_quat.round(5).tolist())

    target, light_err, color_wb = fit_appearance(f, scene, [frames[t] for t in [0, *later]], q_arm,
                                                 robot_exact, robot)

    result = {
        "episode": ep,
        "scene": scene,
        "fit": {
            "rack_iou": round(rack_iou, 3),
            "rack_dxdyyaw": np.round(rack_x, 4).tolist(),
            "slot_occupancy": np.round(occupancy, 3).tolist(),
            "lying_xy_score": [np.round(v, 4).tolist() for v in lying],
            "lying_xyyaw_refined": np.round(lying_xyyaw, 4).tolist(),
            "multiview_loss": [round(loss_before, 4), round(loss_after, 4)],
            "ego_views": ego_t,
            "ego_focal_mm": mv.focal,
            "lying_seen_at_frame": seen_at,
            "absent_vials": missing,
            "mat_iou": round(mat_iou, 3),
            "robot_wb_rgb_sat": color_wb,
            "appearance_rgb": {k: np.round(v, 1).tolist() for k, v in target.items()},
            "lighting_mae_255": round(light_err, 2),
            "settle_shift_m": round(settle_shift, 4),
        },
        "check": check(f, scene, state, action),
        "seconds": round(time.time() - t0, 1),
    }
    if job["overlays"]:
        save_overlay(f, Path(job["overlays"]), ep, img, scene, robot)
    return result


def fit_appearance(f: SceneFitter, scene: dict, frames: list[np.ndarray], q_arm, robot_exact,
                   robot) -> tuple[dict, float, list[float]]:
    """Luz e cor do robô (em ``scene``) com a cena assentada; ``frames[0]`` é o 1º frame."""
    f.env.reset(seed=0, options={"scene": scene})
    f.mj.mj_forward(f.m, f.d)
    masks, target = appearance_target(f, frames)
    scene["lighting"], light_err = fit_lighting(f, masks, target)
    f.place(q_arm)
    floor = (f.labels() == f.m.geom("lightbox_floor").id) & ~robot
    scene["robot_color"], color_wb = robot_color(frames[0], robot_exact, floor)
    return target, light_err, color_wb


def relight(job: dict) -> dict:
    """Linha de ``scenes.jsonl`` com a lightbox em ``spec.DATASET_LAYOUT_OFFSET`` e a luz e a cor
    do robô reajustadas; as poses (já assentadas) não mudam."""
    from replay_dataset import held_interval
    from scipy.ndimage import binary_dilation

    from vials_lite.joint_mapping import lerobot_to_radians, usd_radians_to_mjcf

    global _fitter
    if _fitter is None:
        _fitter = SceneFitter()
    f = _fitter
    t0 = time.time()
    row = json.loads(json.dumps(job["row"]))
    ep, data, scene, fit = row["episode"], Path(job["data"]), row["scene"], row["fit"]
    state, action = load_episode(data, ep)
    held = held_interval(state[:, 5], action[:, 5])
    later = list(range(LATER_FRAMES_STEP, (held[0] if held else len(state)) - 5, LATER_FRAMES_STEP))
    frames = load_frames(data, ep, [0, *later])
    q_arm = usd_radians_to_mjcf(lerobot_to_radians(state[0]))
    f.set_camera(scene)
    f.place(q_arm)
    robot_exact = np.isin(f.labels(), f.robot_geoms)
    robot = binary_dilation(robot_exact, iterations=ROBOT_MARGIN_PX)
    before = {"lighting": scene.get("lighting"), "robot_color": scene.get("robot_color"),
              "lighting_mae_255": fit.get("lighting_mae_255")}
    scene["lightbox_offset"] = list(f.spec.DATASET_LAYOUT_OFFSET)
    target, light_err, color_wb = fit_appearance(f, scene, [frames[t] for t in [0, *later]], q_arm,
                                                 robot_exact, robot)
    fit.update(robot_wb_rgb_sat=color_wb, lighting_mae_255=round(light_err, 2), before_relight=before,
               appearance_rgb={k: np.round(v, 1).tolist() for k, v in target.items()})
    row["seconds_relight"] = round(time.time() - t0, 1)
    return row


def save_overlay(f: SceneFitter, folder: Path, ep: int, img: np.ndarray, scene: dict, robot) -> None:
    """Frame do vídeo com os contornos da cena reconstruída: rack, frascos, tapete; e o render."""
    from PIL import Image

    env = f.env
    env.reset(seed=0, options={"scene": scene})
    f.d.qpos[:] = env.data.qpos
    f.mj.mj_forward(f.m, f.d)
    lab = f.labels()
    out = img.copy()
    for geoms, color in (([f.rack_geom], (255, 0, 0)), ([g for v in f.vial_geoms for g in v], (0, 255, 0)),
                         ([f.ground_geoms[0]], (0, 128, 255))):
        mk = np.isin(lab, geoms)
        edge = mk ^ np.roll(mk, 1, 0) | mk ^ np.roll(mk, 1, 1)
        out[edge] = color
    out[robot & ~np.roll(robot, 1, 0)] = (255, 255, 255)
    rgb = f.mj.Renderer(f.m, HEIGHT, WIDTH)
    rgb.update_scene(f.d, camera="external_D455")
    sim = rgb.render()
    rgb.close()
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate([out, sim], 1)).save(folder / f"ep{ep:03d}.png")


def main() -> None:
    import multiprocessing as mp

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data", type=Path, help="pasta do dataset (data/ e videos/)")
    ap.add_argument("--out", type=Path, required=True, help="arquivo .jsonl de saída (sobrescreve)")
    ap.add_argument("--camera", type=Path, default=Path("outputs/camera_fit/pass2_fov51.5.jsonl"))
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="padrão: os da --camera")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--overlays", type=Path, default=None, help="pasta para as sobreposições")
    ap.add_argument("--relight", type=Path, default=None,
                    help="cenas já reconstruídas (.jsonl): só lightbox, luz e cor do robô")
    args = ap.parse_args()
    if args.relight is not None:
        main_relight(args)
        return

    with args.camera.open(encoding="utf-8") as fh:
        cameras = {r["episode"]: r for r in map(json.loads, fh) if r}
    episodes = sorted(cameras) if args.episodes is None else args.episodes
    jobs = [{"data": str(args.data), "episode": e, "camera": cameras[e],
             "overlays": str(args.overlays) if args.overlays else None} for e in episodes]
    print(f"{len(jobs)} episódios, {args.workers} processos")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(reconstruct, jobs), 1):
            results.append(r)
            fit, chk = r["fit"], r["check"]
            print(f"[{n}/{len(jobs)} {time.time() - t0:4.0f}s] ep {r['episode']:3d}:"
                  f" rack IoU {fit['rack_iou']:.2f},"
                  f" tapete IoU {fit['mat_iou']:.2f} (yaw {r['scene']['mat_yaw']:+.2f}),"
                  f" no rack {r['scene']['vials_in_rack']}, vistos {fit['lying_seen_at_frame']},"
                  f" cor {r['scene']['robot_color']}, agarre a {chk.get('grasp_axis_dist_m')} m do eixo,"
                  f" soltura a {chk.get('release_slot_dist_m')} m do slot", flush=True)
    results.sort(key=lambda r: r["episode"])
    with args.out.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main_relight(args) -> None:
    import multiprocessing as mp

    with args.relight.open(encoding="utf-8") as fh:
        rows = [r for r in map(json.loads, fh) if r]
    if args.episodes is not None:
        rows = [r for r in rows if r["episode"] in args.episodes]
    jobs = [{"data": str(args.data), "row": r} for r in rows]
    print(f"{len(jobs)} episódios, {args.workers} processos")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(relight, jobs), 1):
            results.append(r)
            b, fit = r["fit"]["before_relight"], r["fit"]
            print(f"[{n}/{len(jobs)} {time.time() - t0:4.0f}s] ep {r['episode']:3d}:"
                  f" erro da luz {b['lighting_mae_255']} -> {fit['lighting_mae_255']},"
                  f" cor {b['robot_color']} -> {r['scene']['robot_color']}", flush=True)
    results.sort(key=lambda r: r["episode"])
    with args.out.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
