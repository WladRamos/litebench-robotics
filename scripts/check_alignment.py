"""Confere o alinhamento cinemático entre o USD do robô do workshop e o MJCF do Menagerie.

Monta a cinemática direta do USD a partir das juntas (``localPos/localRot`` de cada
``PhysicsRevoluteJoint``, eixo z), compara com a do MJCF em poses aleatórias e imprime:

* o frame do body ``base`` do MJCF expresso no frame do link ``base`` do USD;
* o erro por corpo (mm, graus) depois de aplicar ``joint_mapping.JOINT_OFFSETS_MJCF_RAD``;
* a pose da base do MJCF no mundo do workshop, a comparar com ``spec.MJCF_BASE_POS/YAW``.

Com ``--episode`` (um parquet do dataset do curso), mostra também onde fica a ponta da garra
(site ``gripperframe``) em alguns frames: no episódio 0, o agarre (~84) deve cair sobre um frasco
e a soltura (~175) sobre o rack. Ler parquet exige pyarrow, que não é dependência do projeto:
use ``uv run --with pyarrow``.

Uso (uma linha; funciona em PowerShell e bash):
    python scripts/check_alignment.py <workshop>/source/sim_to_real_so101/assets/usd/SO-ARM101-USD.usd
        --menagerie assets/third_party/mujoco_menagerie [--episode file-000.parquet]

Com o Menagerie em 4d038b3 e o workshop em ce807d9: erro < 0,002 mm e 0,000° em todos os corpos;
base do MJCF em (−0,06576; 0,02079; 0,03248), yaw 0,0002°.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vials_lite import spec  # noqa: E402
from vials_lite.joint_mapping import (  # noqa: E402
    JOINT_OFFSETS_MJCF_RAD,
    lerobot_to_radians,
    usd_radians_to_mjcf,
)

USD_BODIES = ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw")
MJCF_BODIES = ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")


def _quat_to_mat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _pose(pos, quat_wxyz) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _quat_to_mat(quat_wxyz)
    m[:3, 3] = pos
    return m


def _rot_z(angle: float) -> np.ndarray:
    m = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    m[:2, :2] = [[c, -s], [s, c]]
    return m


class UsdChain:
    """Cadeia serial do USD: corpo filho = pai · localPose0 · Rz(q) · localPose1⁻¹."""

    def __init__(self, usd_path: Path) -> None:
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise RuntimeError(f"não abriu: {usd_path}")
        root = stage.GetDefaultPrim().GetPath()
        self.frames = []
        for name in spec.JOINTS_USD:
            joint = UsdPhysics.Joint(stage.GetPrimAtPath(root.AppendPath(f"joints/{name}")))
            axis = joint.GetPrim().GetAttribute("physics:axis").Get()
            if axis != "Z":
                raise ValueError(f"{name}: eixo {axis}, o cálculo supõe Z")
            r0, r1 = joint.GetLocalRot0Attr().Get(), joint.GetLocalRot1Attr().Get()
            a = _pose(joint.GetLocalPos0Attr().Get(), [r0.GetReal(), *r0.GetImaginary()])
            b = _pose(joint.GetLocalPos1Attr().Get(), [r1.GetReal(), *r1.GetImaginary()])
            self.frames.append((a, np.linalg.inv(b)))

    def bodies(self, q_usd: np.ndarray) -> list[np.ndarray]:
        out, m = [], np.eye(4)
        for (a, b_inv), angle in zip(self.frames, q_usd, strict=True):
            m = m @ a @ _rot_z(angle) @ b_inv
            out.append(m.copy())
        return out


class MjcfChain:
    def __init__(self, menagerie: Path) -> None:
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(menagerie / "robotstudio_so101" / "so101.xml"))
        self.data = mujoco.MjData(self.model)
        self.qpos = [self.model.joint(j).qposadr[0] for j in spec.JOINTS_MENAGERIE]
        self.ids = [self.model.body(b).id for b in MJCF_BODIES]
        self.site = self.model.site("gripperframe").id

    def _fk(self, q_mjcf: np.ndarray) -> None:
        self.data.qpos[self.qpos] = q_mjcf
        self._mj.mj_kinematics(self.model, self.data)

    def bodies(self, q_mjcf: np.ndarray) -> list[np.ndarray]:
        self._fk(q_mjcf)
        return [_pose(self.data.xpos[i], self.data.xquat[i]) for i in self.ids]

    def tip(self, q_mjcf: np.ndarray) -> np.ndarray:
        self._fk(q_mjcf)
        return self.data.site_xpos[self.site].copy()


def _pose_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    dp = np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000.0
    rel = a[:3, :3].T @ b[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)))
    return dp, ang


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("usd", type=Path, help="SO-ARM101-USD.usd do workshop")
    ap.add_argument("--menagerie", type=Path, required=True)
    ap.add_argument("--episode", type=Path, help="parquet de um episódio do dataset (opcional)")
    ap.add_argument("--poses", type=int, default=200)
    args = ap.parse_args()

    usd, mjcf = UsdChain(args.usd), MjcfChain(args.menagerie)
    np.set_printoptions(precision=5, suppress=True)

    # Base do MJCF no frame da base do USD, a partir do ombro com todas as juntas em zero.
    zero = np.zeros(6)
    base_in_usd = usd.bodies(zero)[0] @ np.linalg.inv(mjcf.bodies(usd_radians_to_mjcf(zero))[0])
    print("base do MJCF no frame da base do USD:\n", base_in_usd)

    rng = np.random.default_rng(0)
    lo, hi = np.deg2rad([-110, -100, -100, -95, -160, -10]), np.deg2rad([110, 100, 90, 95, 160, 100])
    worst = np.zeros((len(USD_BODIES), 2))
    for _ in range(args.poses):
        q = rng.uniform(lo, hi)
        pairs = zip(usd.bodies(q), mjcf.bodies(usd_radians_to_mjcf(q)), strict=True)
        errs = np.array([_pose_error(u, base_in_usd @ m) for u, m in pairs])
        worst = np.maximum(worst, errs)
    print(f"\npior erro em {args.poses} poses aleatórias (offsets = {JOINT_OFFSETS_MJCF_RAD}):")
    for name, (dp, ang) in zip(USD_BODIES, worst, strict=True):
        print(f"  {name:10s} {dp:8.4f} mm  {ang:7.4f}°")

    world_usd_base = _rot_z(np.deg2rad(spec.ROBOT_BASE_YAW_DEG))
    world_usd_base[:3, 3] = spec.ROBOT_BASE_POS
    world_mjcf_base = world_usd_base @ base_in_usd
    yaw = np.degrees(np.arctan2(world_mjcf_base[1, 0], world_mjcf_base[0, 0]))
    tilt = np.degrees(np.arccos(np.clip(world_mjcf_base[2, 2], -1.0, 1.0)))
    print(f"\nbase do MJCF no mundo: pos {world_mjcf_base[:3, 3]}  yaw {yaw:.4f}°  inclinação {tilt:.4f}°")
    print(f"spec.MJCF_BASE_POS = {spec.MJCF_BASE_POS}, MJCF_BASE_YAW_DEG = {spec.MJCF_BASE_YAW_DEG}")

    if args.episode:
        import pyarrow.parquet as pq

        table = pq.read_table(args.episode, columns=["observation.state"])
        state = np.array(table.column("observation.state").to_pylist(), dtype=np.float64)
        q_mjcf = usd_radians_to_mjcf(lerobot_to_radians(state))
        print(f"\nponta da garra (gripperframe) no mundo, {len(state)} frames:")
        for f in sorted({0, 84, 100, 140, 175, len(state) - 1} & set(range(len(state)))):
            p = world_mjcf_base @ np.append(mjcf.tip(q_mjcf[f]), 1.0)
            print(f"  frame {f:4d}  garra (LeRobot) {state[f, 5]:5.1f}  ponta {p[:3]}")
        print("  frascos:", [v.pos for v in spec.VIALS], " rack:", spec.RACK.pos)


if __name__ == "__main__":
    main()
