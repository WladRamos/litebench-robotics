"""Especificação da tarefa Vials-to-Rack, extraída do workshop da NVIDIA.

Fonte: https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop (Apache-2.0), commit ``ce807d9``.
Arquivos de origem indicados em cada bloco.

Tudo aqui é **dado**, não lógica: o objetivo é que a versão MuJoCo e o original no Isaac
compartilhem uma única fonte de verdade para tempos, poses, faixas de randomização e o
critério de sucesso. A geometria vem dos assets USD do workshop; o que segue marcado como
hipótese não pôde ser fechado só com os arquivos.

Convenções: metros, radianos, quaternions (w, x, y, z), frame do mundo do Isaac
(z para cima). O robô fica em (-0.05, 0, 0), rotacionado 90° em yaw.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Tempo — tasks/so101_env_cfg.py e tasks/vials_to_rack_env_cfg.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Timing:
    sim_dt: float = 1 / 120  # sim.dt
    decimation: int = 2  # → controle a 60 Hz
    episode_steps: int = 450  # episode_length_s = 450 / 60 (cena -Eval)
    warmup_steps_eval: int = 10  # lerobot_eval.py: primeiros 10 steps usam initial_action
    dataset_fps_label: int = 30  # o que o info.json declara

    @property
    def control_hz(self) -> float:
        return 1.0 / (self.sim_dt * self.decimation)


# Lido no código do workshop: o gravador (scripts/lerobot_agent.py + utils/lerobot_recorder.py)
# grava UM frame por step de 60 Hz, sem subamostrar, mas o dataset é rotulado como 30 fps. Na
# avaliação (utils/lerobot_interface.py, get_action) cada ação do chunk também é aplicada em UM
# step de 60 Hz. Treino e avaliação são coerentes entre si: 1 frame do dataset = 1 step de 60 Hz.
# Os timestamps do dataset não confirmam nem refutam isso: o LeRobot os sintetiza a partir do fps
# declarado. A versão MuJoCo mantém essa semântica para ser comparável.
FRAMES_PER_CONTROL_STEP = 1


# ---------------------------------------------------------------------------
# Robô — assets/so101.py
# ---------------------------------------------------------------------------

# Nomes das juntas: LeRobot (dataset) ↔ USD do workshop ↔ MJCF do MuJoCo Menagerie.
JOINTS_LEROBOT = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
JOINTS_USD = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
JOINTS_MENAGERIE = JOINTS_LEROBOT  # robotstudio_so101/so101.xml usa os nomes do LeRobot

# Pose do link `base` do USD no mundo (init_state do ArticulationCfg; o Isaac a aplica ao link raiz,
# e o yaw de +90° gravado no próprio USD não entra; confirmado com o dataset).
ROBOT_BASE_POS = (-0.05, 0.0, 0.0)
ROBOT_BASE_YAW_DEG = 90.0

# A mesma base no frame do body `base` do MJCF do Menagerie (medida com scripts/check_alignment.py).
# O frame do MJCF é o do USD girado −90° em z e deslocado; com esta pose, os corpos do ombro ao
# punho coincidem com os do USD (< 0,001 mm) para quaisquer ângulos. z = 0,0325 põe o fundo da
# malha da base em z = 0,030, como no USD (o tapete vai de 0,029 a 0,035).
MJCF_BASE_POS = (-0.06576, 0.02079, 0.03248)
MJCF_BASE_YAW_DEG = 0.0

# Pose inicial (rad) e também a initial_action do lerobot_eval.py.
INITIAL_JOINT_POS = (-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, -0.1465)

# Atuadores implícitos (PD) do Isaac: rigidez e amortecimento por junta. effort_limit_sim = 30.
# Não são os mesmos do MJCF do Menagerie (kp=998.22, kv=2.731, forcerange ±2.94). A cena escolhe
# entre os dois pelo parâmetro ``actuators`` de ``vials_lite.scene.build_model``.
ISAAC_ACTUATORS = {
    "Rotation": {"stiffness": 55.0, "damping": 0.7},
    "Pitch": {"stiffness": 30.0, "damping": 0.8},
    "Elbow": {"stiffness": 25.0, "damping": 0.7},
    "Wrist_Pitch": {"stiffness": 12.0, "damping": 0.5},
    "Wrist_Roll": {"stiffness": 7.0, "damping": 0.5},
    "Jaw": {"stiffness": 4.0, "damping": 0.3},
}


# ---------------------------------------------------------------------------
# Objetos e cena — tasks/vials_to_rack_env_cfg.py, tasks/task_env_cfg.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectSpawn:
    name: str
    pos: tuple[float, float, float]
    euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mass: float = 0.0


VIAL_MASS = 0.02  # kg, angular_damping = 100
# O frasco tem physxRigidBody:maxLinearVelocity = 2 no USD (Vial_opaque.usda); m/s.
VIAL_MAX_LINEAR_VELOCITY = 2.0
RACK_MASS = 0.2  # o que o workshop declara (MassPropertiesCfg)
# O que o Isaac simula: o Isaac Lab só aplica o MassPropertiesCfg a prims com
# MassAPI, que o Vial_rack_simple.usda não tem; o PhysX usa então a densidade padrão, 1000 kg/m³,
# sobre o volume da malha (886 cm³, assets/vials_lite/rack.obj).
RACK_MASS_ISAAC = 0.886
VIAL_SPAWN_Z = 0.05

VIALS = (
    ObjectSpawn("vial_1", (0.23, -0.08, VIAL_SPAWN_Z), (0, 90, 0), VIAL_MASS),
    ObjectSpawn("vial_2", (0.23, 0.0, VIAL_SPAWN_Z), (0, 90, 0), VIAL_MASS),
    ObjectSpawn("vial_3", (0.23, -0.16, VIAL_SPAWN_Z), (0, 90, 0), VIAL_MASS),
)
RACK = ObjectSpawn("rack_left", (0.18, 0.08, 0.06), (0, 0, 0), RACK_MASS)
MAT_POS = (0.22, 0.0, 0.032)  # yaw 90°
LIGHTSTUDIO_POS = (-0.1, 0.0, 0.0257)

# lightbox-simple.usd, no frame do LightStudio: caixa de papel
# ("Paper.mdl") com base, topo, fundo (+x) e laterais (±y), aberta em −x (o lado do robô). Fora
# dela o Isaac renderiza preto (o domo tem visible_in_primary_ray=False). Placas de ~4 mm; aqui as
# faces internas. Luz: RectLight 0,30 × 0,01 m no topo, ao longo de x, intensidade 500
# (normalizada), cone 180°, exposição sorteada em RESET.lightbox_exposure (−3..1) a cada reset.
LIGHTBOX_INNER_MIN = (-0.002, -0.379, 0.002)  # x = borda da base (sem parede em −x)
LIGHTBOX_INNER_MAX = (0.502, 0.381, 0.505)
LIGHTBOX_WALL = 0.004
LIGHTBOX_LIGHT_CENTER = (0.2536, 0.0016, 0.5035)  # frame do LightStudio
LIGHTBOX_LIGHT_LENGTH = 0.30  # ao longo de x
# Arranjo do dataset: no vídeo da câmera externa, a lightbox está 5,0 cm mais perto do robô em x
# que no código do workshop (frente do piso 40–60 px fora em todos os episódios; ajuste em grade,
# Δy sem efeito), e a câmera calibrada, o tapete, o rack e os frascos reconstruídos têm o mesmo
# desvio (−4,1 a −4,8 cm). Hipótese não verificada: o dataset foi gravado com o robô em x = 0, não
# em ROBOT_BASE_POS; aqui o robô fica onde está e o resto se desloca de DATASET_LAYOUT_OFFSET
# (x, y, m). Usado na lightbox das cenas reconstruídas (``lightbox_offset``) e no
# ``layout="dataset"`` do env. As variantes eval e dr-eval seguem o código do workshop.
DATASET_LAYOUT_OFFSET = (ROBOT_BASE_POS[0], ROBOT_BASE_POS[1])
# Pose inicial dos operadores no dataset do curso, no espaço do LeRobot (−100..100; garra 0..100):
# mediana de observation.state no frame 0 dos 75 episódios.
# p5–p95: ombro −98,4 a −81,8; cotovelo 64,8 a 97,9; garra 11,9 a 47,1. Braço recolhido e garra
# entreaberta, contra o braço erguido e a garra fechada de INITIAL_JOINT_POS. O motivo de o dataset
# ter sido gravado com outra pose não é conhecido. Usada no ``init_pose="dataset"`` do env.
DATASET_INITIAL_ACTION = (-2.23, -97.58, 82.19, 35.69, -56.53, 24.29)

# Geometria extraída dos USD (scripts/inspect_workshop_assets.py + leitura dos pontos das malhas;
# workshop commit ce807d9). Frames locais dos próprios assets, sem escala.
#
# Frasco (Vial_opaque.usda, o usado pela cena): tubo cônico de centrífuga de 50 mL. Origem na base
# do corpo cilíndrico, eixo +z local ao longo do tubo. Valores do colisor (/Vial/collider, que no
# PhysX usa aproximação SDF). O "extent" autorado no USD está desatualizado (~0,7 mm): não use.
VIAL_BODY_RADIUS = 0.0156  # corpo cilíndrico, z_local ∈ [0, 0.085]
VIAL_CAP_RADIUS = 0.0170  # tampa de rosca, z_local ∈ [0.085, 0.0997]
VIAL_TIP_Z = -0.017  # ponta do cone, z_local ∈ [-0.017, 0], do raio do corpo até VIAL_TIP_RADIUS
VIAL_TIP_RADIUS = 0.0036
VIAL_BODY_TOP_Z = 0.085
VIAL_TOP_Z = 0.0997  # comprimento total ≈ 0.1167 m

# Rack (Vial_rack_simple.usda): corpo x, y ∈ [0, 0.12], z ∈ [-0.006, 0.073] no frame local.
# 4 slots (prims Body1/Mesh/top_01..04), grade 2×2 com passo ~6 cm; entrada em z_local = 0.1.
RACK_BODY_Z = (-0.006, 0.073)
RACK_SLOTS_LOCAL = (
    (0.0298, 0.0299, 0.1),  # top_01
    (0.0902, 0.0299, 0.1),  # top_02
    (0.0902, 0.0901, 0.1),  # top_03
    (0.0308, 0.0901, 0.1),  # top_04
)


# ---------------------------------------------------------------------------
# Randomização no reset — VialsToRackEventCfg + mdp/resets.py (random_asset_pose)
# Amostras uniformes somadas à pose padrão; rotação = quat_padrão * quat_from_euler(r, p, y).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResetRanges:
    rack_pose: dict = field(
        default_factory=lambda: {"x": (-0.04, 0.04), "y": (-0.01, 0.01), "yaw": (-0.5, 0.5)}
    )
    vial_pose: dict = field(
        default_factory=lambda: {
            "x": (-0.04, 0.04),
            "y": (-0.01, 0.01),
            "roll": (-0.3, 0.3),
            "yaw": (0.0, 0.0),
        }
    )
    fixed_vial_z: float = 0.05
    # Com probabilidade 0.33, um frasco sorteado já começa num slot sorteado do rack.
    # Ele NÃO conta como sucesso: o critério exige que o frasco tenha sido agarrado.
    rack_placement_prob: float = 0.33
    # Cena base (TaskEventCfg)
    lightbox_exposure: tuple[float, float] = (-3.0, 1.0)
    mat_yaw: tuple[float, float] = (-0.1, 0.1)
    ego_focal_length_mm: tuple[float, float] = (12.0, 15.0)
    external_cam_pos: dict = field(
        default_factory=lambda: {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.01, 0.01)}
    )
    external_cam_rot: dict = field(
        default_factory=lambda: {"roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05)}
    )


@dataclass(frozen=True)
class DRRanges:
    """Extras das variantes -DR (VialsToRackEventDRCfg)."""

    sky_exposure: tuple[float, float] = (-4.0, 3.0)
    sky_temperature_k: tuple[float, float] = (2500.0, 9500.0)
    mat_yaw: tuple[float, float] = (-0.3, 0.3)
    robot_color: bool = True  # sorteia uma de ROBOT_COLORS a cada reset


@dataclass(frozen=True)
class LightingCalibration:
    """Equivalentes MuJoCo das luzes do workshop (scene.light_colors), com exposição 0.

    O Isaac renderiza com path tracing e tone mapping; o MuJoCo soma luzes e satura em 1. As
    constantes não saem do USD: são ajustadas para que a cor do piso e da parede do fundo no render
    reproduza a dos 75 vídeos do dataset (scripts/fit_lighting.py). No vídeo, a
    exposição sorteada aparece comprimida (o piso varia ~1,6× entre p5 e p95, não 16×): o fator de
    cada luz é 2 ** (exposure_scale · exposição), com a mesma compressão para as duas luzes (o tone
    mapping é global). O dataset é todo da variante DR; para a ``eval`` (sem domo), a premissa é
    que a fita sozinha, com exposição 0, deixa o piso no brilho mediano do vídeo (não verificável:
    o dataset não tem episódios dessa variante).
    """

    # Ajuste de scripts/fit_lighting.py com as cores medidas nas cenas reconstruídas (máscaras com
    # o tapete na pose de cada episódio, para não misturar tapete e piso): piso p5/50/95 = 131/174/213 (vídeo 130/175/211), parede do fundo 163/205/251
    # (157/215/234), R/B do piso 1,00/1,05/1,26 (0,98/1,06/1,26); piso na eval 173 (premissa).
    rect_diffuse: float = 0.3176  # soma das luzes da fita
    rect_ambient: float = 0.4763
    dome_diffuse: float = 0.4323
    dome_ambient: float = 0.0
    exposure_scale: float = 0.2115
    temperature_strength: float = 2.3053  # 0 = domo branco; 1 = cor do corpo negro relativa a white_k
    white_k: float = 9091.6  # temperatura que sai branca
    dome_elevation_deg: float = 10.48  # a luz do domo entra pela frente aberta, descendo este ângulo


LIGHTING = LightingCalibration()


# mdp/resets.py (randomize_robot_color): diffuse_color_constant do material "3D printed" do robô.
# A variante sem DR usa a cor do USD, laranja (é a dos episódios laranja do dataset).
ROBOT_COLORS = {
    "orange": (0.876, 0.317, 0.132),
    "teal": (0.0, 0.8, 0.502),
    "white": (0.95, 0.95, 0.95),
    "black": (0.08, 0.08, 0.08),
}
DEFAULT_ROBOT_COLOR = "orange"


# ---------------------------------------------------------------------------
# Câmeras — tasks/task_env_cfg.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraSpec:
    name: str  # chave no dataset: observation.images.<name>
    width: int = 640
    height: int = 480
    focal_length_mm: float = 13.5
    horizontal_aperture_mm: float | None = None  # None = não definido no workshop
    vertical_aperture_mm: float | None = None
    # Pose no mundo (só câmeras fixas). Convenção de câmera USD/OpenGL = a do MuJoCo:
    # olha para -z local, +y local para cima.
    pos: tuple[float, float, float] | None = None
    quat_wxyz: tuple[float, float, float, float] | None = None
    # FOV vertical em graus quando calibrado contra o dataset (senão, sai da focal e da abertura)
    fovy_deg: float | None = None
    note: str = ""


CAMERAS = (
    CameraSpec(
        "ego",
        note="presa em Robot/gripper/gripper_cam; offset pos (-0.005, 0.06, -0.062), "
        "rot euler (-45, 0, 0) graus, convenção OpenGL. Focal 13,5 mm ('10th of real'). A abertura "
        "não é definida no workshop: vale o padrão do PinholeCameraCfg do Isaac Lab (suposto)",
    ),
    # Em task_env_cfg.py esta câmera tem spawn=None: pose e intrínseca vêm
    # do payload remoto Isaac/5.1/Isaac/Sensors/Intel/RealSense/rsd455.usd (prim
    # RSD455/Camera_OmniVision_OV9782_Right), NÃO dos 13,5 mm do camera_object. Pose composta:
    # LightStudio (-0.1, 0, 0.0257) ∘ LightBox ∘ camera_mount ∘ RSD455 ∘ câmera.
    # Extraído: pos (0.1376, 0.3297, 0.4886), FOV horizontal 2·atan(3.896 / (2·1.93)) ≈ 90,5°.
    # Esses valores NÃO reproduzem o vídeo do dataset. Calibrados contra os 75
    # episódios (scripts/fit_external_camera.py): FOV vertical 51,5° (IQR por
    # episódio 49–53°) e posição = extraída + (−0,0473; −0,0069; −0,0115) m, a mediana dos
    # deslocamentos; a orientação mediana difere < 0,2° da extraída e foi mantida. A origem dessa
    # diferença no Isaac não foi identificada (hipótese: FOV horizontal = 64,9°, o das aberturas na
    # vertical → 51,0° na vertical).
    CameraSpec(
        "external_D455",
        focal_length_mm=1.93,
        horizontal_aperture_mm=3.896,
        vertical_aperture_mm=2.453,
        pos=(0.0903, 0.3228, 0.4771),
        quat_wxyz=(-0.0599, -0.0278, 0.3056, 0.9499),
        fovy_deg=51.5,
        note="pose padrão calibrada no dataset, antes da randomização do camera_mount "
        "(ResetRanges.external_cam_*)",
    ),
)

# Texto da tarefa. Padrão: o que a avaliação em simulação do curso passa às políticas
# (``--lang_instruction`` em "Sim Evaluation", nas duas variantes) e o robô real também usa.
LANGUAGE_INSTRUCTION = "Pick up the vial and place it in the yellow rack"
# O único texto do dataset do curso (``meta/tasks.parquet``), o que as políticas viram no treino;
# é também o padrão do ``--lang_instruction`` no ``lerobot_eval.py``.
DATASET_TASK = "Pick up the vial and place it in the rack"


# ---------------------------------------------------------------------------
# Sucesso — mdp/terms.py (vial_placed_on_rack + vial_placed_on_rack_termination)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuccessParams:
    warmup_steps: int = 30
    grasp_history_window: int = 20
    grasp_force_threshold_n: float = 2.0  # contato jaw↔frasco
    grasp_min_height: float = 0.055  # só para o evento [GRASP] (observação), não para o sucesso
    rack_local_x: tuple[float, float] = (0.0, 0.12)
    rack_local_y: tuple[float, float] = (0.0, 0.12)
    rack_local_z_max: float = 0.1
    vertical_threshold: float = 0.7  # |eixo z do frasco · z do mundo|
    confirm_steps: int = 25  # condições válidas por 25 steps seguidos → terminated


TIMING = Timing()
RESET = ResetRanges()
DR = DRRanges()
SUCCESS = SuccessParams()
