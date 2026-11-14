"""Montagem da cena Vials-to-Rack em MuJoCo a partir de ``spec.py``.

Robô: MJCF do MuJoCo Menagerie (``robotstudio_so101/so101.xml``), contatos originais, base em
``spec.MJCF_BASE_POS`` (alinhada à do USD do workshop por cinemática direta). Atuadores por preset:
``"isaac"`` (padrão: o PD do workshop, que reproduz as trajetórias do dataset) ou ``"menagerie"``
(servos STS3215 do MJCF).

Objetos, com as aproximações que a versão MuJoCo exige (o workshop usa colisores SDF das malhas;
no MuJoCo uma malha vira o seu fecho convexo, e os furos do rack sumiriam):

* frasco: dois cilindros (corpo Ø 31,2 mm, tampa Ø 34 mm) e a ponta em cone convexo
  (``VIAL_TIP_SHAPE``), 20 g, amortecimento angular D = c·I nos graus rotacionais, com o c = 100 do
  ``angular_damping`` do workshop;
* rack: placa de fundo + por furo, anéis de 16 caixas: assento cônico (de r 5 mm em z_local −0,002
  a r 15 mm em 0,014, onde a ponta do frasco encaixa), parede reta (raio 17 mm, até 0,051) e
  chanfro (até raio 23 mm em 0,073) + tampo plano e cantos, medidos na malha do USD, com a massa que o
  Isaac simula (``RACK_MASS_MODE``);
* aparência do rack e do frasco: malhas só visuais (sem contato e sem massa) derivadas dos USD
  (``assets/vials_lite/``, ``scripts/export_workshop_meshes.py``); a colisão fica no grupo
  ``HIDDEN_GROUP``, que o ``Renderer`` não desenha. Cores dos USD convertidas de linear para sRGB
  (``linear_to_srgb``), como o Isaac mostra. O rack leva a oclusão de ambiente gravada na malha
  (``AO_MATERIAL``, ``scripts/bake_ambient_occlusion.py``): sem ela, os furos ficam claros, e no
  Isaac são escuros;
* contato: atrito 0,5 em todos os geoms, o material padrão do Isaac Lab (``SimulationCfg.physics_material``,
  que vale para todo corpo sem material próprio, caso de todos os USD da cena); a garra do Menagerie
  (atrito 1 com torção e rolamento, ``condim`` 6) passa a 0,5 com torção 0,005 e sem rolamento
  (``condim`` 4). Sem torção, o frasco pendula na garra em torno do eixo de pinça, e no Isaac fica
  preso: o contato SDF do PhysX cobre uma área, e as primitivas do Menagerie tocam em poucos pontos;
* tapete: caixa do ``mat.usda`` (0,457 × 0,305 × 0,006 m, yaw 90°), estático;
* chão: plano em z = 0,029 (fundo do tapete), onde se apoiam o tapete e a base do robô; não é
  desenhado (grupo ``HIDDEN_GROUP``). O piso visível é o da lightbox, em z = 0,0277.

Câmeras (convenção OpenGL = a do MuJoCo):

* ``ego``: no body ``gripper`` (o frame é o mesmo do USD), offset (-0,005; 0,06; -0,062) e
  rotação −45° em x, como no workshop. FOV vertical pela focal (13,5 mm; o reset sorteia 12–15) e
  pela abertura horizontal padrão do Isaac Lab, 20,955 mm (suposta: o workshop não a define;
  ver ``spec.CAMERAS``);
* ``external_D455``: pose e FOV vertical (51,5°) de ``spec.CAMERAS``, calibrados contra o vídeo
  dos 75 episódios do dataset (``scripts/fit_external_camera.py``), e não os extraídos do USD.

Lightbox (``spec.LIGHTBOX_*``): paredes e piso só visuais, aberta em −x, com fundo preto fora dela
como no Isaac. Luzes em ``set_lighting``: a fita da lightbox, o domo das variantes DR e um ambiente,
com intensidades calibradas contra o vídeo do dataset (``spec.LIGHTING``, ``scripts/fit_lighting.py``).
Cores e texturas dos objetos continuam aproximadas.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import spec

GROUND_Z = 0.029
MAT_HALF = (0.1524, 0.2286, 0.003)  # mat.usda (0,2286 × 0,1524 × 0,003) com yaw 90°: x/y trocados
VIAL_ANGULAR_DAMPING = 100.0  # 1/s, angular_damping do RigidObjectCfg dos frascos (usado em "physx")
VIAL_IMPLICIT_DAMPING = 100.0  # 1/s, c do D = c·I em "implicit"
# Como aplicar o amortecimento angular dos frascos:
# * "implicit" (padrão): D = c·I no dof_damping, com c = VIAL_IMPLICIT_DAMPING, o mesmo c do workshop.
#   Com a torção da garra (GRIPPER_TORSION), reproduz o Isaac no replay de ações gravadas lá.
# * "physx": a fórmula do PhysX (computeUnconstrainedVelocity), ω ← ω·max(0, 1 − c·dt) antes do solver,
#   com c = VIAL_ANGULAR_DAMPING, feita por ``physics_step``. É fiel ao código do PhysX, mas no MuJoCo
#   freia mais (sobra 1/6 por passo) e o frasco quase não tomba depois de solto. Fica como experimento.
VIAL_DAMPING_MODES = ("implicit", "physx")
VIAL_DAMPING_MODE = "implicit"
# Colisor da ponta do frasco:
# * "cone" (padrão): cone convexo (malha de VIAL_TIP_SEGMENTS lados) com a inclinação do USD até
#   spec.VIAL_TIP_RADIUS, aparado no topo (VIAL_TIP_CLEARANCE). Desliza pelo chanfro do furo e se centra,
#   como no Isaac: no replay das ações gravadas lá, o frasco entra no furo nos mesmos episódios.
# * "cylinder": um cilindro de fundo chato com raio 8 mm, a aproximação mais simples; com ele, o frasco
#   solto sobre o furo fica sobre o tampo a partir de ~22 mm de deslocamento.
# Massa do rack: "isaac" (padrão), spec.RACK_MASS_ISAAC, a que o Isaac de fato simula; "declared",
# spec.RACK_MASS, a que o workshop declara.
RACK_MASS_MODES = ("declared", "isaac")
RACK_MASS_MODE = "isaac"
# Limita a velocidade linear dos frascos a spec.VIAL_MAX_LINEAR_VELOCITY depois de cada passo, como o
# maxLinearVelocity do PhysX.
VIAL_VELOCITY_LIMIT = True
VIAL_TIP_SHAPES = ("cylinder", "cone")
VIAL_TIP_SHAPE = "cone"
VIAL_TIP_SEGMENTS = 24
# O anel de cima do cone fica 0,4 mm para dentro do corpo: com o raio igual, o frasco deitado no tapete
# apoia nos vértices da malha e no cilindro do corpo ao mesmo tempo e rola devagar (7,5° em 10 s).
VIAL_TIP_INSET = 0.0004
# Topo do cone aparado: None, o cone inteiro (do anel acima até a ponta); um valor, a inclinação do USD só
# até o raio do corpo menos isto e, dali até z = 0, um cilindro desse raio. O contato mole deixa o frasco
# afundar 2–4 mm no tapete quando a garra o empurra; com o cone inteiro, a face inclinada toca o tapete e
# o frasco escapa da pinça. Escondido atrás do corpo, o topo não muda a silhueta. Os 3 mm compensam o
# contato mole do MuJoCo, que o Isaac (rígido) não tem.
VIAL_TIP_CLEARANCE = 0.003
# RigidBodyMaterialCfg padrão do Isaac Lab: estático = dinâmico = 0,5, combinação por média
ISAAC_FRICTION = 0.5
# Atrito de torção da garra (geom_friction[:, 1], condim 4): segura o frasco como o contato SDF do Isaac.
# Menos que isso o deixa pendular; 0,005 é o menor valor de uma varredura que basta.
GRIPPER_TORSION = 0.005

# Rack (malha de Vial_rack_simple.usda; frame local, origem no canto).
RACK_SIZE = 0.12
# Fundo do furo: assento cônico de (r 5 mm, z −0,002) a (r 15 mm, z 0,014), aberto no centro, onde
# a ponta cônica do frasco encaixa. Um piso plano em 0,014 deixaria o frasco no slot ~1,6 cm mais
# alto que no Isaac.
RACK_SEAT_BOTTOM = (0.005, -0.002)
RACK_SEAT_TOP_RADIUS = 0.015
RACK_FLOOR_TOP = 0.014
RACK_BORE_RADIUS = 0.017
RACK_BORE_TOP = 0.051  # fim do furo reto, começo do chanfro
RACK_TOP = 0.073
RACK_CHAMFER_RADIUS = 0.023  # raio do chanfro no topo
RACK_WALL_SEGMENTS = 16
# Tampo plano em volta do chanfro (r de 23 a 30 mm, face de cima em RACK_TOP). Sem ele, o que fica ali
# é a face de cima das caixas inclinadas do chanfro, uma rampa que desce até ~1,6 mm para fora, onde
# a ponta do frasco escorrega; no USD, o tampo é plano.
RACK_FLAT_TOP = True
RACK_FLAT_TOP_OUTER = 0.030  # meia célula: até onde o anel vai, ao longo dos eixos
RACK_FLAT_TOP_HALF_Z = 0.002
# Borda do furo como no USD: (r, z) da superfície, da parede reta ao tampo, medidos na malha
# (rack.obj, mediana de 12 ângulos, iguais nos 4 furos). Usado quando RACK_HOLE_PROFILE o recebe;
# None (padrão): o chanfro reto de (RACK_BORE_RADIUS, RACK_BORE_TOP) a (RACK_CHAMFER_RADIUS, RACK_TOP).
RACK_HOLE_PROFILE_USD = (
    (0.0170, 0.0510), (0.0172, 0.0545), (0.0174, 0.0562), (0.0178, 0.0589), (0.0184, 0.0622),
    (0.0192, 0.0655), (0.0200, 0.0680), (0.0210, 0.0702), (0.0220, 0.0718), (0.0230, 0.0727),
    (0.0240, 0.0730),
)
RACK_HOLE_PROFILE = None

HIDDEN_GROUP = 3  # grupo de geoms que o Renderer não desenha (padrão: grupos 0–2)
LIGHTBOX_PAPER_ALBEDO = 0.9
# Cores dos USD (diffuse_color_constant, lineares; convertidas por linear_to_srgb ao montar a cena)
RACK_LINEAR = (0.9382, 0.5865, 0.0435)  # Vial_rack_simple.usda, OmniPBR
VIAL_CAP_LINEAR = (0.1216, 0.3222, 0.8138)  # Vial_opaque.usda, Vial_cap
# Corpo do frasco: "plastic_white" (vMaterials, translúcido), sem cor no USD; cinza claro (já em sRGB)
VIAL_BODY_RGB = (0.74, 0.72, 0.72)
# Calibração contra o vídeo: ganho sobre a cor sRGB do USD para que a razão objeto/piso no render
# (mediana dos pixels, frame 0 dos 75 episódios) case com a do vídeo. O corpo do frasco e o tapete,
# sem cor no USD, já vão calibrados acima e abaixo.
RACK_VIDEO_GAIN = (1.0, 1.0, 0.74)
VIAL_CAP_VIDEO_GAIN = (1.32, 1.30, 1.13)
MAT_RGB = (0.115, 0.106, 0.097)  # Rubber_Smooth (mat.usda), sem cor no USD
VISUAL_MESHES = ("rack", "vial_body", "vial_cap")
# Oclusão de ambiente gravada nas malhas (scripts/bake_ambient_occlusion.py): a fração do céu
# visível vai em u e uma rampa de cinza multiplica a cor (0 → AO_FLOOR, 1 → 1)
AO_MATERIAL = "baked_ao"
AO_RAMP_TEXELS = 256
AO_UV_MARGIN = 0.05  # a textura repete: u fica longe das bordas para não misturar 0 com 1
AO_FLOOR = 0.0  # 0,3 deixava os furos claros demais (scripts/offline_gap.py um pouco pior)
# Serrilha da tampa como textura: a malha é um cilindro liso com u em períodos de
# dente (scripts/export_workshop_meshes.py, knurl_u). Um período tem KNURL_TEXELS texels e, na faixa
# KNURL_GROOVE (fração do período), o sulco escurece a cor por KNURL_GROOVE_SHADE. O MuJoCo usa
# mipmap: na câmera externa (dente < 1 pixel) a tampa sai na média, sem moiré; de perto, na câmera
# de punho, os sulcos aparecem como as linhas escuras finas do Isaac.
KNURL_MATERIAL = "knurl"
KNURL_TEXELS = 32
KNURL_GROOVE = (0.5, 0.65)
KNURL_GROOVE_SHADE = 0.5
LIGHT_SEGMENTS = 3  # spots ao longo da fita da RectLight
SHADOW_MAP_SIZE = 8192
# plano de corte distante, em unidades de stat.extent (~1,5 m): o padrão do MuJoCo (50) estica o
# mapa de sombra até ~74 m e a falta de precisão de profundidade vira pontilhado e vãos na sombra
ZFAR_EXTENTS = 2.0
ISAAC_DEFAULT_HORIZONTAL_APERTURE_MM = 20.955  # PinholeCameraCfg do Isaac Lab
EGO_POS = (-0.005, 0.06, -0.062)
EGO_EULER_X_DEG = -45.0
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480


def ego_fovy_deg(focal_length_mm: float) -> float:
    """FOV vertical da câmera de punho em 640×480 (abertura vertical = horizontal · 480/640)."""
    vertical_aperture = ISAAC_DEFAULT_HORIZONTAL_APERTURE_MM * IMAGE_HEIGHT / IMAGE_WIDTH
    return float(np.degrees(2.0 * np.arctan(vertical_aperture / (2.0 * focal_length_mm))))


def default_menagerie_dir() -> Path:
    """``$VIALS_LITE_MENAGERIE`` ou ``assets/third_party/mujoco_menagerie`` do repositório."""
    env = os.environ.get("VIALS_LITE_MENAGERIE")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "assets" / "third_party" / "mujoco_menagerie"


def assets_dir() -> Path:
    """``assets/vials_lite`` do repositório: malhas visuais derivadas dos USD do workshop."""
    return Path(__file__).resolve().parents[2] / "assets" / "vials_lite"


def linear_to_srgb(rgb) -> np.ndarray:
    """Cor linear (como ``diffuse_color_constant`` do Isaac) → sRGB, o que o MuJoCo mostra direto.

    O Isaac aplica a curva sRGB na imagem final; sem isto as cores saem escuras e saturadas.
    """
    c = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def video_color(linear_rgb, gain) -> np.ndarray:
    """Cor do USD (linear) em sRGB vezes o ganho calibrado contra o vídeo, limitada a 1."""
    return np.clip(linear_to_srgb(linear_rgb) * np.asarray(gain, dtype=float), 0.0, 1.0)


def robot_rgb(color: str) -> np.ndarray:
    """Cor de ``spec.ROBOT_COLORS`` (linear, como no workshop) para o ``mat_rgba`` do MuJoCo."""
    return linear_to_srgb(spec.ROBOT_COLORS[color])


def _quat_axis_angle(axis, angle: float) -> list[float]:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return [float(np.cos(angle / 2)), *(np.sin(angle / 2) * axis).tolist()]


def mesh_material(mesh: str) -> str:
    """Material da malha visual: serrilha na tampa, oclusão gravada onde houver ``vt``, ou nenhum."""
    if mesh == "vial_cap":
        return KNURL_MATERIAL
    return AO_MATERIAL if has_baked_ao(mesh) else ""


@functools.cache
def has_baked_ao(mesh: str) -> bool:
    """A malha traz a oclusão de ambiente gravada nas coordenadas de textura (``vt``)."""
    with (assets_dir() / f"{mesh}.obj").open(encoding="utf-8") as f:
        return any(line.startswith("vt ") for line in f)


def ao_to_u(ao):
    """Fração do céu visível (0..1) → coordenada u da rampa de ``AO_MATERIAL``."""
    return AO_UV_MARGIN + (1.0 - 2.0 * AO_UV_MARGIN) * np.asarray(ao, dtype=float)


def _ao_ramp(floor: float) -> bytes:
    """Textura cinza ``AO_RAMP_TEXELS`` × 1: em u = ``ao_to_u(ao)``, brilho ``floor``..1 (linear)."""
    u = (np.arange(AO_RAMP_TEXELS) + 0.5) / AO_RAMP_TEXELS
    ao = np.clip((u - AO_UV_MARGIN) / (1.0 - 2.0 * AO_UV_MARGIN), 0.0, 1.0)
    g = np.round(255 * (floor + (1.0 - floor) * ao)).astype(np.uint8)
    return np.repeat(g[:, None], 3, 1).tobytes()


def _knurl_texture(groove: tuple[float, float], shade: float) -> bytes:
    """Um período de dente (``KNURL_TEXELS`` × 1): branco, e brilho ``shade`` na faixa do sulco."""
    u = (np.arange(KNURL_TEXELS) + 0.5) / KNURL_TEXELS
    g = np.where((u >= groove[0]) & (u < groove[1]), round(255 * shade), 255).astype(np.uint8)
    return np.repeat(g[:, None], 3, 1).tobytes()


def _add_visual_mesh(body, mesh: str, rgba) -> None:
    """Geom só visual (sem contato e sem massa) com uma malha de ``assets_dir()``."""
    import mujoco

    body.add_geom(
        name=f"{body.name}_{mesh}",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh,
        material=mesh_material(mesh),
        rgba=list(rgba),
        contype=0,
        conaffinity=0,
        density=0.0,
    )


def _add_vial_tip_mesh(spec_mj) -> None:
    """Malha do cone da ponta (``VIAL_TIP_SHAPE == "cone"``): o anel de cima em z = 0 (e, com
    ``VIAL_TIP_CLEARANCE``, o do começo da inclinação) e o da ponta."""
    th = 2 * np.pi * np.arange(VIAL_TIP_SEGMENTS) / VIAL_TIP_SEGMENTS
    if VIAL_TIP_CLEARANCE is None:
        rings = [(spec.VIAL_BODY_RADIUS - VIAL_TIP_INSET, 0.0), (spec.VIAL_TIP_RADIUS, spec.VIAL_TIP_Z)]
    else:
        r_top = spec.VIAL_BODY_RADIUS - VIAL_TIP_CLEARANCE
        slope = spec.VIAL_TIP_Z / (spec.VIAL_BODY_RADIUS - spec.VIAL_TIP_RADIUS)
        z_top = slope * (spec.VIAL_BODY_RADIUS - r_top)
        rings = [(r_top, 0.0), (r_top, z_top), (spec.VIAL_TIP_RADIUS, spec.VIAL_TIP_Z)]
    pts = [np.stack([r * np.cos(th), r * np.sin(th), np.full_like(th, z)], 1) for r, z in rings]
    vert = np.concatenate(pts)
    spec_mj.add_mesh(name="vial_tip_cone", uservert=vert.ravel().tolist())


def _add_vial(spec_mj, name: str) -> None:
    """Colisão invisível em cilindros, com a ponta em cone (``VIAL_TIP_SHAPE``); aparência pelas malhas do
    USD."""
    import mujoco

    if VIAL_TIP_SHAPE not in VIAL_TIP_SHAPES:
        raise ValueError(f"VIAL_TIP_SHAPE deve ser um de {VIAL_TIP_SHAPES}, não {VIAL_TIP_SHAPE!r}")
    body = spec_mj.worldbody.add_body(name=name, pos=[0.0, 0.0, 0.0])
    body.add_freejoint(name=name)
    cyl = mujoco.mjtGeom.mjGEOM_CYLINDER
    hb = spec.VIAL_BODY_TOP_Z / 2
    hc = (spec.VIAL_TOP_Z - spec.VIAL_BODY_TOP_Z) / 2
    ht = -spec.VIAL_TIP_Z / 2
    hidden = dict(group=HIDDEN_GROUP, rgba=[0.85, 0.9, 1.0, 1.0])
    body.add_geom(
        name=f"{name}_body",
        type=cyl,
        size=[spec.VIAL_BODY_RADIUS, hb, 0],
        pos=[0, 0, hb],
        mass=0.016,
        **hidden,
    )
    body.add_geom(
        name=f"{name}_cap",
        type=cyl,
        size=[spec.VIAL_CAP_RADIUS, hc, 0],
        pos=[0, 0, spec.VIAL_BODY_TOP_Z + hc],
        mass=0.003,
        **hidden,
    )
    if VIAL_TIP_SHAPE == "cylinder":
        tip = dict(type=cyl, size=[0.008, ht, 0], pos=[0, 0, -ht])
    else:
        tip = dict(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="vial_tip_cone")
    body.add_geom(name=f"{name}_tip", mass=0.001, **tip, **hidden)
    _add_visual_mesh(body, "vial_body", [*VIAL_BODY_RGB, 1.0])
    _add_visual_mesh(body, "vial_cap", [*video_color(VIAL_CAP_LINEAR, VIAL_CAP_VIDEO_GAIN), 1.0])


def _add_rack(spec_mj) -> None:
    """Colisão do rack em caixas (o fecho convexo da malha taparia os furos), invisível.

    Por furo, de baixo para cima: assento cônico (fundo do furo, onde a ponta do frasco encaixa),
    parede reta e chanfro, cada um um anel de ``RACK_WALL_SEGMENTS`` caixas cuja face interna segue
    o perfil medido na malha do USD; mais o tampo plano, os cantos da célula e uma placa de fundo.
    """
    import mujoco

    box = mujoco.mjtGeom.mjGEOM_BOX
    body = spec_mj.worldbody.add_body(name=spec.RACK.name, pos=list(spec.RACK.pos))
    body.add_freejoint(name=spec.RACK.name)
    z0 = spec.RACK_BODY_Z[0]
    geoms: list[dict] = [  # placa de fundo, embaixo do assento
        dict(
            size=[RACK_SIZE / 2, RACK_SIZE / 2, (RACK_SEAT_BOTTOM[1] - z0) / 2],
            pos=[0.06, 0.06, (RACK_SEAT_BOTTOM[1] + z0) / 2],
        )
    ]
    n = RACK_WALL_SEGMENTS
    plate_top = RACK_SEAT_BOTTOM[1]
    # (r, z) da face interna embaixo e em cima, meias-espessuras radial e tangencial de cada anel.
    # Só a placa de fundo toca o chão: o resto começa em cima dela, e o assento, inclinado, é fino
    # para não passar por baixo da placa.
    if RACK_HOLE_PROFILE is None:
        rim = ((RACK_BORE_RADIUS, RACK_BORE_TOP), (RACK_CHAMFER_RADIUS, RACK_TOP))
    else:
        rim = tuple(RACK_HOLE_PROFILE)
    rings = (
        (RACK_SEAT_BOTTOM, (RACK_SEAT_TOP_RADIUS, RACK_FLOOR_TOP), 0.003, 0.0025),
        ((RACK_BORE_RADIUS, plate_top), (RACK_BORE_RADIUS, RACK_BORE_TOP), 0.0065, 0.012),
        *((a, b, 0.0065, 0.012) for a, b in zip(rim[:-1], rim[1:], strict=False)),
    )
    flat_inner = rim[-1][0]  # o tampo plano começa onde a borda chega ao tampo
    for cx, cy, _ in spec.RACK_SLOTS_LOCAL:
        for k in range(n):
            theta = 2 * np.pi * k / n
            radial = np.array([np.cos(theta), np.sin(theta)])
            q_yaw = np.array(_quat_axis_angle([0, 0, 1], theta))
            for (ra, za), (rb, zb), radial_half, tangential_half in rings:
                # caixa girada em torno do eixo tangencial, para fora: a face interna vai de (ra, za)
                # a (rb, zb) no plano radial, e a caixa fica do lado de fora dela
                tilt = np.arctan2(rb - ra, zb - za)
                x_axis = (np.cos(tilt), -np.sin(tilt))  # x local após o giro: (radial, z)
                c_r = (ra + rb) / 2 + radial_half * x_axis[0]
                c_z = (za + zb) / 2 + radial_half * x_axis[1]
                q = np.zeros(4)
                mujoco.mju_mulQuat(q, q_yaw, np.array(_quat_axis_angle([0, 1, 0], tilt)))
                geoms.append(
                    dict(
                        size=[radial_half, tangential_half, np.hypot(rb - ra, zb - za) / 2],
                        pos=[cx + c_r * radial[0], cy + c_r * radial[1], c_z],
                        quat=q.tolist(),
                    )
                )
            if RACK_FLAT_TOP:
                # tampa plana do segmento: a corda interna fica em r = flat_inner, sem invadir a borda
                r_mid = (flat_inner + RACK_FLAT_TOP_OUTER) / 2
                geoms.append(
                    dict(
                        size=[(RACK_FLAT_TOP_OUTER - flat_inner) / 2, r_mid * np.tan(np.pi / n) * 1.05,
                              RACK_FLAT_TOP_HALF_Z],
                        pos=[cx + r_mid * radial[0], cy + r_mid * radial[1], RACK_TOP - RACK_FLAT_TOP_HALF_Z],
                        quat=q_yaw.tolist(),
                    )
                )
        # Cantos da célula 6 × 6 cm, fora do raio do chanfro.
        for sx in (-1, 1):
            for sy in (-1, 1):
                geoms.append(
                    dict(
                        size=[0.006, 0.006, (RACK_TOP - plate_top) / 2],
                        pos=[cx + sx * 0.024, cy + sy * 0.024, (RACK_TOP + plate_top) / 2],
                    )
                )
    if RACK_MASS_MODE not in RACK_MASS_MODES:
        raise ValueError(f"RACK_MASS_MODE deve ser um de {RACK_MASS_MODES}, não {RACK_MASS_MODE!r}")
    mass = (spec.RACK_MASS if RACK_MASS_MODE == "declared" else spec.RACK_MASS_ISAAC) / len(geoms)
    for i, g in enumerate(geoms):
        # colisão invisível: os segmentos apareceriam serrilhados; a aparência vem da malha visual
        body.add_geom(
            name=f"rack_{i}", type=box, mass=mass, group=HIDDEN_GROUP, rgba=[0.95, 0.8, 0.1, 1], **g
        )
    _add_visual_mesh(body, "rack", [*video_color(RACK_LINEAR, RACK_VIDEO_GAIN), 1.0])


@dataclass(frozen=True)
class SceneIds:
    """Índices do modelo compilado usados pelo ambiente."""

    arm_qpos: np.ndarray
    arm_qvel: np.ndarray
    actuators: np.ndarray
    vial_bodies: np.ndarray
    vial_qpos: np.ndarray
    vial_qvel: np.ndarray
    vial_ang_dofs: np.ndarray  # graus rotacionais que ``physics_step`` amortece (vazio em "implicit")
    rack_body: int
    rack_qpos: int
    rack_qvel: int
    jaw_body: int
    gripper_site: int
    mat_geom: int
    robot_materials: np.ndarray  # peças impressas (não os motores): recebem spec.ROBOT_COLORS
    ego_cam: int
    external_cam: int
    rect_lights: np.ndarray  # spots da fita da lightbox
    lightbox_geoms: np.ndarray  # paredes e piso da lightbox
    dome_light: int
    fill_light: int


ACTUATOR_PRESETS = ("menagerie", "isaac")
ISAAC_EFFORT_LIMIT = 30.0  # effort_limit_sim do ImplicitActuatorCfg (assets/so101.py)
# Armature das juntas do braço no preset "isaac". O USD não tem armature, mas com zero a garra (12 g,
# kp 4) não segura o frasco de forma estável no MuJoCo: demo roteirizada em 9/30. 0,005 é o padrão
# da classe ``so101`` do Menagerie e leva a demo a 27/30, sem perder a fidelidade no replay das
# trajetórias do dataset (p95 ≤ 1,5° no braço).
ISAAC_PRESET_ARMATURE = 0.005


def _apply_isaac_friction(m) -> None:
    """Atrito do Isaac em todos os geoms; na garra (``priority`` 1 no Menagerie, cujos parâmetros
    prevalecem no contato), torção ``GRIPPER_TORSION`` e sem rolamento (``condim`` 4)."""
    m.geom_friction[:, 0] = ISAAC_FRICTION
    gripper = m.geom_priority == 1
    m.geom_friction[gripper, 1] = GRIPPER_TORSION
    m.geom_friction[gripper, 2] = 0.0
    m.geom_condim[gripper] = 4


def _apply_isaac_actuators(s) -> None:
    """Troca os servos do Menagerie pelo PD implícito do workshop (``spec.ISAAC_ACTUATORS``).

    O ``ImplicitActuatorCfg`` só define rigidez, amortecimento e limite de esforço; atrito e
    amortecimento de junta não estão autorados no USD (PhysX: zero) e são zerados em ``build_model``,
    depois da compilação. O amortecimento do PD entra como ``kv`` do atuador, tratado implicitamente
    pelo integrador ``implicitfast``. O armature também é zero no USD, mas aqui fica em
    ``ISAAC_PRESET_ARMATURE`` (ver a constante).
    """
    for mj_name, usd_name in zip(spec.JOINTS_MENAGERIE, spec.JOINTS_USD, strict=True):
        g = spec.ISAAC_ACTUATORS[usd_name]
        kp, kd = g["stiffness"], g["damping"]
        a = s.actuator(mj_name)
        a.gainprm[0] = kp
        a.biasprm[0:3] = [0.0, -kp, -kd]
        a.forcerange = [-ISAAC_EFFORT_LIMIT, ISAAC_EFFORT_LIMIT]


def _add_lightbox(spec_mj) -> None:
    """Paredes, piso e luzes da lightbox (``spec.LIGHTBOX_*``), só visuais.

    Luzes (intensidades em ``set_lighting``): a RectLight vira ``LIGHT_SEGMENTS`` luzes ao longo da
    fita, apontando para baixo: spots com cone de 90° e expoente 1 (≈ emissor lambertiano), menos a
    do meio, direcional e a única que projeta sombra; ``dome`` é uma direcional que entra pela
    frente aberta, como a luz do domo das variantes DR; ``fill`` só tem ambiente (o rebote no
    papel, que o MuJoCo não calcula). A headlight do MuJoCo fica desligada: ela acompanharia a
    câmera.
    """
    import mujoco

    world = spec_mj.worldbody
    origin = np.array(spec.LIGHTSTUDIO_POS)
    lo, hi, w = np.array(spec.LIGHTBOX_INNER_MIN), np.array(spec.LIGHTBOX_INNER_MAX), spec.LIGHTBOX_WALL
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    panels = {  # nome: (centro, meia-extensão) no frame do LightStudio
        "floor": ((mid[0], mid[1], lo[2] - w / 2), (half[0], half[1], w / 2)),
        "top": ((mid[0], mid[1], hi[2] + w / 2), (half[0], half[1], w / 2)),
        "back": ((hi[0] + w / 2, mid[1], mid[2]), (w / 2, half[1] + w, half[2] + w)),
        "left": ((mid[0], hi[1] + w / 2, mid[2]), (half[0], w / 2, half[2] + w)),
        "right": ((mid[0], lo[1] - w / 2, mid[2]), (half[0], w / 2, half[2] + w)),
    }
    for name, (center, size) in panels.items():
        world.add_geom(
            name=f"lightbox_{name}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(size),
            pos=list(origin + center),
            rgba=[LIGHTBOX_PAPER_ALBEDO] * 3 + [1.0],
            contype=0,
            conaffinity=0,
        )
    spec_mj.visual.headlight.active = 0
    spec_mj.visual.quality.shadowsize = SHADOW_MAP_SIZE
    spec_mj.visual.map.zfar = ZFAR_EXTENTS
    center = origin + spec.LIGHTBOX_LIGHT_CENTER
    n = LIGHT_SEGMENTS
    for k in range(n):
        dx = spec.LIGHTBOX_LIGHT_LENGTH * ((k + 0.5) / n - 0.5)
        world.add_light(
            name=f"lightbox_{k}",
            pos=list(center + [dx, 0.0, -0.001]),
            dir=[0, 0, -1],
            # o do meio é direcional e projeta a sombra: a sombra de spot (projeção em perspectiva)
            # sai pontilhada nas peças finas do robô; a direcional (ortográfica) não
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL if k == n // 2 else mujoco.mjtLightType.mjLIGHT_SPOT,
            cutoff=90.0,
            exponent=1.0,
            castshadow=k == n // 2,
        )
    dome_dir = dome_direction(spec.LIGHTING.dome_elevation_deg)
    world.add_light(
        name="dome",
        pos=list(center - 0.8 * dome_dir),
        dir=list(dome_dir),
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        castshadow=False,
    )
    world.add_light(
        name="fill",
        pos=list(center),
        dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        castshadow=False,
    )


def dome_direction(elevation_deg: float) -> np.ndarray:
    """Direção da luz do domo: entra pela frente aberta (−x → +x), descendo ``elevation_deg``."""
    a = np.deg2rad(elevation_deg)
    return np.array([np.cos(a), 0.0, -np.sin(a)])


def blackbody_rgb(kelvin: float) -> np.ndarray:
    """Cor (RGB linear, máximo 1) de um corpo negro, aproximação de Tanner Helland (1000–40000 K)."""
    t = np.clip(kelvin, 1000.0, 40000.0) / 100.0
    r = 1.0 if t <= 66 else np.clip(329.698727446 * (t - 60) ** -0.1332047592 / 255, 0, 1)
    if t <= 66:
        g = np.clip((99.4708025861 * np.log(t) - 161.1195681661) / 255, 0, 1)
    else:
        g = np.clip(288.1221695283 * (t - 60) ** -0.0755148492 / 255, 0, 1)
    if t >= 66:
        b = 1.0
    elif t <= 19:
        b = 0.0
    else:
        b = np.clip((138.5177312231 * np.log(t - 10) - 305.0447927307) / 255, 0, 1)
    rgb = np.array([r, g, b], dtype=float)
    return rgb / rgb.max()


def light_colors(
    rect_exposure: float,
    dome_exposure: float | None = None,
    temperature_k: float = 6500.0,
    calib: spec.LightingCalibration = spec.LIGHTING,
) -> dict[str, np.ndarray]:
    """Diffuse e ambiente (RGB) de cada grupo de luzes para as exposições (stops) do workshop.

    ``dome_exposure`` None = sem domo (variante ``eval``). O fator de cada luz é
    2 ** (exposure_scale · exposição) (``spec.LightingCalibration``). A cor do domo é a do corpo
    negro em ``temperature_k`` relativa a ``calib.white_k``, misturada ao branco por
    ``calib.temperature_strength``.
    """
    gain = 2.0 ** (calib.exposure_scale * rect_exposure)
    out = {
        "rect": np.full(3, calib.rect_diffuse * gain),
        "dome": np.zeros(3),
        "ambient": np.full(3, calib.rect_ambient * gain),
    }
    if dome_exposure is not None:
        tint = blackbody_rgb(temperature_k) / blackbody_rgb(calib.white_k)
        # temperature_strength > 1 exagera a cor (o ajuste pede isso); nenhum canal fica negativo
        tint = np.maximum(1.0 + calib.temperature_strength * (tint / tint.mean() - 1.0), 0.0)
        dome = 2.0 ** (calib.exposure_scale * dome_exposure) * tint
        out["dome"] = calib.dome_diffuse * dome
        out["ambient"] = out["ambient"] + calib.dome_ambient * dome
    return out


def set_lighting(
    model,
    ids: SceneIds,
    rect_exposure: float = 0.0,
    dome_exposure: float | None = None,
    temperature_k: float = 6500.0,
) -> None:
    """Aplica ``light_colors`` às luzes do modelo compilado."""
    c = light_colors(rect_exposure, dome_exposure, temperature_k)
    model.light_diffuse[ids.rect_lights] = c["rect"] / len(ids.rect_lights)
    model.light_specular[ids.rect_lights] = 0.1 * c["rect"] / len(ids.rect_lights)
    model.light_ambient[ids.rect_lights] = 0.0
    model.light_diffuse[ids.dome_light] = c["dome"]
    model.light_specular[ids.dome_light] = 0.1 * c["dome"]
    model.light_ambient[ids.dome_light] = 0.0
    model.light_diffuse[ids.fill_light] = 0.0
    model.light_specular[ids.fill_light] = 0.0
    model.light_ambient[ids.fill_light] = c["ambient"]


def build_model(menagerie_dir: Path | None = None, actuators: str = "isaac"):
    """Compila a cena; devolve ``(MjModel, SceneIds)``.

    actuators: "isaac" (padrão; PD do workshop, ``spec.ISAAC_ACTUATORS``) ou "menagerie" (servos
    STS3215 do MJCF: kp 998, forcerange ±2,94). O "isaac" reproduz as trajetórias do dataset do
    curso (``scripts/replay_dynamic.py``); o "menagerie" fica para quem quiser o servo do robô real.
    """
    import mujoco

    if actuators not in ACTUATOR_PRESETS:
        raise ValueError(f"actuators deve ser um de {ACTUATOR_PRESETS}, não {actuators!r}")

    menagerie_dir = Path(menagerie_dir) if menagerie_dir is not None else default_menagerie_dir()
    xml = menagerie_dir / "robotstudio_so101" / "so101.xml"
    if not xml.is_file():
        raise FileNotFoundError(
            f"MJCF do Menagerie não encontrado em {xml}. "
            "Veja assets/README.md ou defina VIALS_LITE_MENAGERIE."
        )
    s = mujoco.MjSpec.from_file(str(xml))
    s.option.timestep = spec.TIMING.sim_dt
    s.visual.global_.offwidth = IMAGE_WIDTH
    s.visual.global_.offheight = IMAGE_HEIGHT

    s.body("base").pos = list(spec.MJCF_BASE_POS)
    s.body("base").quat = _quat_axis_angle([0, 0, 1], np.deg2rad(spec.MJCF_BASE_YAW_DEG))
    if actuators == "isaac":
        _apply_isaac_actuators(s)

    for name in VISUAL_MESHES:
        path = assets_dir() / f"{name}.obj"
        if not path.is_file():
            raise FileNotFoundError(f"{path} não encontrado: rode scripts/export_workshop_meshes.py")
        # shell: a malha é só visual (massa zero), mas o compilador calcula a inércia mesmo assim
        s.add_mesh(name=name, file=str(path), inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
    tex = s.add_texture(
        name=AO_MATERIAL, type=mujoco.mjtTexture.mjTEXTURE_2D, width=AO_RAMP_TEXELS, height=1, nchannel=3
    )
    tex.data = _ao_ramp(AO_FLOOR)
    s.add_material(name=AO_MATERIAL, textures=["", AO_MATERIAL])
    tex = s.add_texture(
        name=KNURL_MATERIAL, type=mujoco.mjtTexture.mjTEXTURE_2D, width=KNURL_TEXELS, height=1, nchannel=3
    )
    tex.data = _knurl_texture(KNURL_GROOVE, KNURL_GROOVE_SHADE)
    s.add_material(name=KNURL_MATERIAL, textures=["", KNURL_MATERIAL])
    world = s.worldbody
    world.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1, 1, 0.01],
        pos=[0, 0, GROUND_Z],
        group=HIDDEN_GROUP,
    )
    _add_lightbox(s)
    world.add_geom(
        name="mat",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(MAT_HALF),
        pos=list(spec.MAT_POS),
        rgba=[*MAT_RGB, 1],
    )
    if VIAL_TIP_SHAPE == "cone":
        _add_vial_tip_mesh(s)
    for v in spec.VIALS:
        _add_vial(s, v.name)
    _add_rack(s)

    ego_quat = _quat_axis_angle([1, 0, 0], np.deg2rad(EGO_EULER_X_DEG))
    s.body("gripper").add_camera(name="ego", pos=list(EGO_POS), quat=ego_quat, fovy=ego_fovy_deg(13.5))
    ext = next(c for c in spec.CAMERAS if c.name == "external_D455")
    world.add_camera(name=ext.name, pos=list(ext.pos), quat=list(ext.quat_wxyz), fovy=ext.fovy_deg)

    m = s.compile()
    names = [m.mat(k).name for k in range(m.nmat)]
    robot_materials = np.array(
        [k for k, n in enumerate(names) if "so101" in n and not n.startswith("sts3215")]
    )
    m.mat_rgba[robot_materials, :3] = robot_rgb(spec.DEFAULT_ROBOT_COLOR)
    _apply_isaac_friction(m)
    if VIAL_DAMPING_MODE not in VIAL_DAMPING_MODES:
        raise ValueError(f"VIAL_DAMPING_MODE deve ser um de {VIAL_DAMPING_MODES}, não {VIAL_DAMPING_MODE!r}")
    vial_ang_dofs = []
    for v in spec.VIALS:
        adr = m.jnt_dofadr[m.joint(v.name).id]
        if VIAL_DAMPING_MODE == "implicit":
            m.dof_damping[adr + 3 : adr + 6] = VIAL_IMPLICIT_DAMPING * m.body_inertia[m.body(v.name).id]
        else:
            vial_ang_dofs += range(adr + 3, adr + 6)
    if actuators == "isaac":
        arm_dofs = [m.joint(j).dofadr[0] for j in spec.JOINTS_MENAGERIE]
        m.dof_damping[arm_dofs] = 0.0
        m.dof_frictionloss[arm_dofs] = 0.0
        m.dof_armature[arm_dofs] = ISAAC_PRESET_ARMATURE

    ids = SceneIds(
        arm_qpos=np.array([m.joint(j).qposadr[0] for j in spec.JOINTS_MENAGERIE]),
        arm_qvel=np.array([m.joint(j).dofadr[0] for j in spec.JOINTS_MENAGERIE]),
        actuators=np.array([m.actuator(j).id for j in spec.JOINTS_MENAGERIE]),
        vial_bodies=np.array([m.body(v.name).id for v in spec.VIALS]),
        vial_qpos=np.array([m.joint(v.name).qposadr[0] for v in spec.VIALS]),
        vial_qvel=np.array([m.joint(v.name).dofadr[0] for v in spec.VIALS]),
        vial_ang_dofs=np.array(vial_ang_dofs, dtype=int),
        rack_body=m.body(spec.RACK.name).id,
        rack_qpos=int(m.joint(spec.RACK.name).qposadr[0]),
        rack_qvel=int(m.joint(spec.RACK.name).dofadr[0]),
        jaw_body=m.body("moving_jaw_so101_v1").id,
        gripper_site=m.site("gripperframe").id,
        mat_geom=m.geom("mat").id,
        robot_materials=robot_materials,
        ego_cam=m.camera("ego").id,
        external_cam=m.camera("external_D455").id,
        rect_lights=np.array([m.light(f"lightbox_{k}").id for k in range(LIGHT_SEGMENTS)]),
        lightbox_geoms=np.array([g for g in range(m.ngeom) if m.geom(g).name.startswith("lightbox_")]),
        dome_light=m.light("dome").id,
        fill_light=m.light("fill").id,
    )
    set_lighting(m, ids)
    return m, ids


def physics_step(m, d, ids: SceneIds) -> None:
    """Um passo de ``sim.dt``. No modo ``"physx"``, amortece antes a rotação dos frascos como o PhysX.

    O PhysX escala a velocidade angular por ``max(0, 1 − c·dt)`` antes do solver (com c = 100 e
    dt = 1/120, sobra 1/6); o solver de contato e a integração vêm depois, com a velocidade já
    amortecida. Aqui, o mesmo: escala o ``qvel`` e chama ``mj_step``. No padrão, ``"implicit"``,
    ``ids.vial_ang_dofs`` é vazio e isto é só o ``mj_step``.

    Com ``VIAL_VELOCITY_LIMIT``, a velocidade linear de cada frasco (mundo, os 3 primeiros dofs do
    freejoint) é reescalada para no máximo ``spec.VIAL_MAX_LINEAR_VELOCITY`` depois do passo.
    """
    import mujoco

    if ids.vial_ang_dofs.size:
        d.qvel[ids.vial_ang_dofs] *= 1.0 - min(1.0, VIAL_ANGULAR_DAMPING * m.opt.timestep)
    mujoco.mj_step(m, d)
    if VIAL_VELOCITY_LIMIT:
        for adr in ids.vial_qvel:
            v = d.qvel[adr : adr + 3]
            speed = float(np.linalg.norm(v))
            if speed > spec.VIAL_MAX_LINEAR_VELOCITY:
                v *= spec.VIAL_MAX_LINEAR_VELOCITY / speed
