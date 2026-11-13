"""Replay cinemático do dataset do curso no MJCF do Menagerie.

Para cada episódio (um parquet por episódio no formato LeRobot v3.0), converte
``observation.state`` para radianos do MJCF (``joint_mapping``), confere as faixas das juntas do
MJCF e mede onde fica a ponta da garra (site ``gripperframe``) no agarre e na soltura, com a base
em ``spec.MJCF_BASE_POS``.

Agarre e soltura saem da própria garra: o frasco está preso enquanto o estado da garra fica mais de
4 unidades acima do comando (o frasco a bloqueia em ~18). O trecho mais longo em que isso acontece
vai do agarre à soltura.

As posições de frascos e rack no dataset não são gravadas (só juntas), e o reset randomiza as duas.
Por isso a comparação é com as poses nominais do ``spec``: serve para conferir o alinhamento e a
escala, não a posição exata de cada episódio.

Ler parquet exige pyarrow, que não é dependência do projeto. Uso (uma linha; PowerShell e bash):
    uv run --with pyarrow python scripts/replay_dataset.py <pasta com file-*.parquet>
        --menagerie assets/third_party/mujoco_menagerie

Os parquets ficam em huggingface.co/datasets/sreetz-nv/so101_teleop_vials_rack_left,
``data/chunk-000/file-000.parquet`` a ``file-074.parquet`` (~1,6 MB no total). Não são versionados
aqui: o dataset não declara licença. São 75 episódios e 18 250 frames.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_alignment import MjcfChain, _rot_z  # noqa: E402

from vials_lite import spec  # noqa: E402
from vials_lite.joint_mapping import lerobot_to_radians, usd_radians_to_mjcf  # noqa: E402

HELD_MARGIN = 4.0  # estado − comando da garra (unidades do LeRobot) acima do qual o frasco está preso


def held_interval(gripper_state: np.ndarray, gripper_action: np.ndarray) -> tuple[int, int] | None:
    """(início, fim) do trecho mais longo com a garra bloqueada; None se não houver."""
    held = (gripper_state - gripper_action) > HELD_MARGIN
    best, start = None, None
    for i, h in enumerate([*held, False]):
        if h and start is None:
            start = i
        elif not h and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def _describe(label: str, pts: np.ndarray) -> None:
    print(f"{label} (n = {len(pts)}):")
    for axis, col in zip("xyz", pts.T, strict=True):
        p5, med, p95 = np.percentile(col, [5, 50, 95])
        print(f"  {axis}: min {col.min():.3f}  p5 {p5:.3f}  mediana {med:.3f}  p95 {p95:.3f}", end="")
        print(f"  max {col.max():.3f}")


def main() -> None:
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path, help="pasta com os file-*.parquet do dataset")
    ap.add_argument("--menagerie", type=Path, required=True)
    args = ap.parse_args()

    mjcf = MjcfChain(args.menagerie)
    world = _rot_z(np.deg2rad(spec.MJCF_BASE_YAW_DEG))
    world[:3, 3] = spec.MJCF_BASE_POS
    model = mjcf.model
    limits = model.jnt_range[[model.joint(j).id for j in spec.JOINTS_MENAGERIE]]

    n_frames, out_frames, worst_deg = 0, np.zeros(6, dtype=int), np.zeros(6)
    grasps, releases, locked, missing = [], [], [], []
    files = sorted(args.data.glob("file-*.parquet"))
    for path in files:
        table = pq.read_table(path, columns=["observation.state", "action", "episode_index"])
        state = np.array(table.column("observation.state").to_pylist(), dtype=np.float64)
        action = np.array(table.column("action").to_pylist(), dtype=np.float64)
        episode = table.column("episode_index")[0].as_py()

        q = usd_radians_to_mjcf(lerobot_to_radians(state))
        n_frames += len(q)
        excess = np.maximum(limits[:, 0] - q, q - limits[:, 1])
        out_frames += (excess > 1e-4).sum(axis=0)
        worst_deg = np.maximum(worst_deg, np.degrees(excess.max(axis=0)))

        interval = held_interval(state[:, 5], action[:, 5])
        if interval is None:
            missing.append(episode)
            continue
        start, end = interval
        end = min(end, len(q) - 1)
        grasps.append((world @ np.append(mjcf.tip(q[start]), 1.0))[:3])
        releases.append((world @ np.append(mjcf.tip(q[end]), 1.0))[:3])
        locked.append(state[start:end, 5].mean())

    print(f"{len(files)} episódios, {n_frames} frames")
    print("faixas das juntas no MJCF (frames fora por mais de 1e-4 rad, pior excesso):")
    for name, n, w in zip(spec.JOINTS_MENAGERIE, out_frames, worst_deg, strict=True):
        print(f"  {name:14s} {n:5d} frames  {max(w, 0.0):.3f}°")
    print(f"sem agarre detectado: {missing or 'nenhum'}")
    if not grasps:
        return
    grasps, releases = np.array(grasps), np.array(releases)
    _describe("\nponta da garra no agarre", grasps)
    _describe("ponta da garra na soltura", releases)
    lock_range = f"{min(locked):.1f}–{max(locked):.1f}"
    print(f"garra travada pelo frasco (LeRobot): mediana {np.median(locked):.1f}, {lock_range}")

    vials = np.array([v.pos[:2] for v in spec.VIALS])
    d_vial = np.linalg.norm(grasps[:, None, :2] - vials[None], axis=2)
    print(
        f"\nagarre → frasco nominal mais próximo {[v.name for v in spec.VIALS]}: "
        f"{np.bincount(d_vial.argmin(1), minlength=len(vials)).tolist()}; "
        f"distância xy mediana {np.median(d_vial.min(1)):.3f} m, máxima {d_vial.min(1).max():.3f} m"
    )
    slots = np.array(spec.RACK.pos[:2]) + np.array([s[:2] for s in spec.RACK_SLOTS_LOCAL])
    d_slot = np.linalg.norm(releases[:, None, :2] - slots[None], axis=2)
    print(
        f"soltura → slot nominal mais próximo (top_01..04): "
        f"{np.bincount(d_slot.argmin(1), minlength=len(slots)).tolist()}; "
        f"distância xy mediana {np.median(d_slot.min(1)):.3f} m, máxima {d_slot.min(1).max():.3f} m"
    )


if __name__ == "__main__":
    main()
