"""Conversão entre o espaço de juntas do LeRobot (dataset) e radianos do simulador.

Porta fiel de ``LeRobotSO101Interface`` (utils/lerobot_interface.py do workshop da NVIDIA):

* juntas 1–5 no LeRobot: [-100, 100] → normalizado [0, 1]
* garra no LeRobot: [0, 100] → normalizado [0, 1]
* normalizado → graus por faixa da junta (``SO101_USD_MAPPING``) → radianos

É um mapeamento **afim** que imita uma calibração; não são ângulos físicos do braço real.
Manter exatamente esta função é o que torna o dataset do curso (75 demos) e os checkpoints
treinados nele utilizáveis na versão MuJoCo. Se o zero das juntas do MJCF diferir do USD,
a correção entra em ``JOINT_OFFSETS_MJCF_RAD`` (medida: só o wrist_roll, +2,789°).
"""

from __future__ import annotations

import numpy as np

# graus, ordem de spec.JOINTS_LEROBOT
JOINT_RANGES_DEG = np.array(
    [
        [-110.0, 110.0],  # shoulder_pan
        [-100.0, 100.0],  # shoulder_lift
        [-100.0, 90.0],  # elbow_flex
        [-95.0, 95.0],  # wrist_flex
        [-160.0, 160.0],  # wrist_roll
        [-10.0, 100.0],  # gripper
    ],
    dtype=np.float64,
)

# Diferença de zero entre o USD do workshop e o MJCF do Menagerie (rad): q_mjcf = q_usd + offset.
# Medida por cinemática direta (scripts/check_alignment.py): só o wrist_roll difere,
# 2,789° (o quaternion do body `gripper` no MJCF tem esse giro a mais). Confere com as faixas:
# a do MJCF (−2,7438..2,8412) é ±160° deslocada de 0,0487.
JOINT_OFFSETS_MJCF_RAD = np.array([0.0, 0.0, 0.0, 0.0, 0.04868, 0.0], dtype=np.float64)


def lerobot_to_radians(values: np.ndarray) -> np.ndarray:
    """LeRobot (6,) ou (N, 6) → radianos do simulador (convenção do USD do workshop)."""
    v = np.asarray(values, dtype=np.float64)
    norm = np.empty_like(v)
    norm[..., :5] = (v[..., :5] + 100.0) / 200.0
    norm[..., 5] = v[..., 5] / 100.0
    lo, hi = JOINT_RANGES_DEG[:, 0], JOINT_RANGES_DEG[:, 1]
    deg = lo + norm * (hi - lo)
    return np.deg2rad(deg)


def radians_to_lerobot(radians: np.ndarray) -> np.ndarray:
    """Inversa de :func:`lerobot_to_radians`."""
    r = np.asarray(radians, dtype=np.float64)
    deg = np.rad2deg(r)
    lo, hi = JOINT_RANGES_DEG[:, 0], JOINT_RANGES_DEG[:, 1]
    norm = (deg - lo) / (hi - lo)
    out = np.empty_like(norm)
    out[..., :5] = norm[..., :5] * 200.0 - 100.0
    out[..., 5] = norm[..., 5] * 100.0
    return out


def usd_radians_to_mjcf(radians: np.ndarray) -> np.ndarray:
    """Radianos na convenção do USD → convenção do MJCF (aplica offsets de zero)."""
    return np.asarray(radians, dtype=np.float64) + JOINT_OFFSETS_MJCF_RAD


def mjcf_radians_to_usd(radians: np.ndarray) -> np.ndarray:
    return np.asarray(radians, dtype=np.float64) - JOINT_OFFSETS_MJCF_RAD
