"""Converte o registro do gravador do Isaac (``scripts/isaac_recorder.py``) para o evalkit.

Entrada: a pasta ``<variante>.rec`` (``meta.json``, ``episodes.jsonl``, ``ep_NNNN.npz``). Saídas:

* ``--run``: um run no formato do ``evalkit.runner`` (``run.json``, ``episodes.jsonl``, ``steps/``,
  ``report.md``), com fases, falhas e fechamentos da garra calculados pelo evalkit sobre o estado do
  Isaac. A ponta da garra (``gripper_pos``, o site ``gripperframe`` do MJCF) sai da cinemática direta
  do MuJoCo com as juntas do Isaac: o alinhamento MJCF ↔ USD foi conferido assim (``check_alignment.py``).
  O episódio *i* ganha a semente ``1000 + i``, a mesma que o runner dá à cena *i* de ``--scenes``:
  ``paired_compare.py`` pareia o run do Isaac com o do Vials-Lite nas mesmas cenas.
* ``--scenes``: um ``scenes.jsonl`` com a cena inicial de cada episódio (``env.SCENE_KEYS``), para o
  ``evalkit.runner --scenes``. A câmera externa recebe o mesmo desvio que o ``camera_mount`` teve no
  Isaac, aplicado à câmera calibrada daqui, como no sorteio da ``eval``; a focal da câmera de
  punho e as luzes vêm do USD. O HDRI do domo não tem equivalente (o domo daqui é uniforme).

Uso (bash; no PowerShell, a mesma linha):
    uv run python scripts/isaac_rec_convert.py outputs/isaac/act_lite/Eval.rec \\
        --run outputs/evalkit/isaac/act_lite_Eval --scenes outputs/isaac/act_lite/Eval.scenes.jsonl \\
        --name act_lite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evalkit.progress import _over_rack_box, _rack_local  # noqa: E402
from evalkit.runner import isaac_joint_order, summarize  # noqa: E402
from evalkit.stats import wilson_interval  # noqa: E402
from vials_lite import spec  # noqa: E402
from vials_lite.joint_mapping import radians_to_lerobot, usd_radians_to_mjcf  # noqa: E402
from vials_lite.success import quat_rotate  # noqa: E402

SEED_BASE = 1000  # a semente que o runner dá à cena 0 de --scenes (--seed padrão)
# ordem das juntas do LeRobot (e do MJCF): os nomes do USD são os mesmos
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
UPRIGHT_MIN = 0.7  # |eixo do frasco · z|, o vertical_threshold do workshop


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def joint_order(names: list[str]) -> list[int]:
    """Índices das juntas do Isaac na ordem do LeRobot (nomes do USD ou do LeRobot)."""
    return isaac_joint_order(names)


def vials_in_rack(vial_pos: np.ndarray, vial_quat: np.ndarray, rack_pos, rack_quat) -> list[int]:
    """Frascos em pé sobre a caixa do rack (os que o reset do workshop põe no rack)."""
    out = []
    for i, (p, q) in enumerate(zip(vial_pos, vial_quat, strict=True)):
        up = abs(quat_rotate(np.asarray(q), np.array([0.0, 0.0, 1.0]))[2])
        if up > UPRIGHT_MIN and _over_rack_box(_rack_local(np.asarray(p), np.asarray(rack_pos),
                                                           np.asarray(rack_quat))):
            out.append(i)
    return out


def robot_color_name(rgb: list[float] | None) -> str:
    """Nome da cor mais próxima em ``spec.ROBOT_COLORS`` (as do workshop)."""
    if rgb is None:
        return spec.DEFAULT_ROBOT_COLOR
    return min(spec.ROBOT_COLORS, key=lambda k: float(np.sum((np.array(spec.ROBOT_COLORS[k]) - rgb) ** 2)))


def yaw(q: list[float]) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class Kinematics:
    """Cinemática direta do MJCF: juntas do Isaac (rad, USD) → ponta da garra (``gripperframe``)."""

    def __init__(self) -> None:
        import mujoco

        from vials_lite.scene import build_model

        self._mj = mujoco
        self.model, self.ids = build_model()
        self.data = mujoco.MjData(self.model)
        self.ext_cam_pos = self.model.cam_pos[self.ids.external_cam].copy()
        self.ext_cam_quat = self.model.cam_quat[self.ids.external_cam].copy()

    def gripper_pos(self, joint_pos_usd: np.ndarray) -> np.ndarray:
        out = np.empty((len(joint_pos_usd), 3))
        for k, q in enumerate(joint_pos_usd):
            self.data.qpos[self.ids.arm_qpos] = usd_radians_to_mjcf(q)
            self._mj.mj_kinematics(self.model, self.data)
            out[k] = self.data.site_xpos[self.ids.gripper_site]
        return out


def scene_row(row: dict, kin: Kinematics, order: list[int], mount_default: dict | None) -> dict:
    """Linha de ``scenes.jsonl`` a partir da cena inicial gravada."""
    s0, usd = row["scene"]["state0"], row["scene"]["usd"]
    vial_pos, vial_quat = np.array(s0["vial_pos"]), np.array(s0["vial_quat"])
    scene: dict = {
        "joint_pos": radians_to_lerobot(np.array(s0["joint_pos"])[order]).tolist(),
        "rack_pos": s0["rack_pos"], "rack_quat": s0["rack_quat"],
        "vial_pos": vial_pos.tolist(), "vial_quat": vial_quat.tolist(),
        "vials_in_rack": vials_in_rack(vial_pos, vial_quat, s0["rack_pos"], s0["rack_quat"]),
        "robot_color": robot_color_name(usd.get("robot_rgb")),
    }
    if usd.get("mat_orient") is not None:
        scene["mat_yaw"] = yaw(usd["mat_orient"]) - np.pi / 2  # o workshop soma 90° ao sorteio
    if usd.get("ego_focal_mm"):
        from vials_lite.scene import ego_fovy_deg

        scene["ego_fovy"] = ego_fovy_deg(float(usd["ego_focal_mm"]))
    if mount_default and usd.get("ext_mount_translate") and usd.get("ext_mount_orient"):
        d = next(iter(mount_default.values()))
        dpos = np.array(usd["ext_mount_translate"]) - np.array(d["pos"])
        dquat = quat_mul(quat_conj(np.array(d["quat"])), np.array(usd["ext_mount_orient"]))
        scene["external_cam_pos"] = (kin.ext_cam_pos + dpos).tolist()
        scene["external_cam_quat"] = quat_mul(kin.ext_cam_quat, dquat).tolist()
    lighting = {}
    if usd.get("rect_exposure") is not None:
        lighting["rect_exposure"] = float(usd["rect_exposure"])
    if usd.get("dome_exposure") is not None:
        lighting["dome_exposure"] = float(usd["dome_exposure"])
        lighting["temperature_k"] = float(usd["dome_temperature_k"])
    if lighting:
        scene["lighting"] = lighting
    return {"episode": row["episode"], "source": "isaac", "dome_texture": usd.get("dome_texture"),
            "scene": scene}


def convert_episode(row: dict, ep: dict, kin: Kinematics, order: list[int]) -> tuple[dict, dict]:
    """Resumo do evalkit e arrays por step (formato de ``steps/ep_NNN.npz`` do runner)."""
    s0 = row["scene"]["state0"]
    ignore = vials_in_rack(np.array(s0["vial_pos"]), np.array(s0["vial_quat"]), s0["rack_pos"],
                           s0["rack_quat"])
    ignore_vial = ignore if ignore else None
    joints = ep["joint_pos"][:, order]
    n = len(joints)
    arrays = {
        "agent_pos": radians_to_lerobot(joints).astype(np.float32),
        "action": radians_to_lerobot(ep["action"][:, order]).astype(np.float32),
        "phase": np.zeros(n, np.int8),
        "gripper_pos": kin.gripper_pos(joints),
        "vial_pos": ep["vial_pos"], "vial_quat": ep["vial_quat"],
        "rack_pos": ep["rack_pos"], "rack_quat": ep["rack_quat"],
        "jaw_contact_force": ep["jaw_force"],
    }
    usd = row["scene"]["usd"]
    summary = {
        "seed": SEED_BASE + row["episode"], "success": row["success"], "steps": n,
        **summarize(arrays, row["success"], ignore_vial),
        "vial_on_rack_at_reset": ignore_vial,
        "robot_color": robot_color_name(usd.get("robot_rgb")),
        "lighting": {k: usd[k] for k in ("rect_exposure", "dome_exposure", "dome_temperature_k")
                     if usd.get(k) is not None},
        "events": [], "wall_s": row.get("wall_s"), "isaac_episode": row["episode"],
        "truncated": row["truncated"], "interrupted": row.get("interrupted", False),
    }
    if usd.get("mat_orient") is not None:
        summary["mat_yaw"] = yaw(usd["mat_orient"]) - np.pi / 2
    return summary, arrays


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rec", type=Path, help="pasta <variante>.rec do gravador")
    ap.add_argument("--run", type=Path, default=None, help="saída: run no formato do evalkit")
    ap.add_argument("--scenes", type=Path, default=None, help="saída: scenes.jsonl para o Vials-Lite")
    ap.add_argument("--name", default=None, help="política, no relatório (padrão: a pasta acima do registro)")
    args = ap.parse_args()
    if not (args.run or args.scenes):
        ap.error("dê --run e/ou --scenes")
    meta = json.loads((args.rec / "meta.json").read_text(encoding="utf-8"))
    lines = (args.rec / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    rows = [r for r in rows if not r.get("interrupted")]
    order = joint_order(meta["joint_names"])
    kin = Kinematics()
    mount_default = next((r["scene"].get("camera_mount_default") for r in rows
                          if r["scene"].get("camera_mount_default")), None)

    if args.scenes:
        args.scenes.parent.mkdir(parents=True, exist_ok=True)
        with args.scenes.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(scene_row(r, kin, order, mount_default)) + "\n")
        print(f"{len(rows)} cenas em {args.scenes}")

    if args.run:
        (args.run / "steps").mkdir(parents=True, exist_ok=True)
        episodes = []
        with (args.run / "episodes.jsonl").open("w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                ep = dict(np.load(args.rec / f"ep_{r['episode']:04d}.npz"))
                summary, arrays = convert_episode(r, ep, kin, order)
                summary["episode"] = i
                np.savez_compressed(args.run / "steps" / f"ep_{i:03d}.npz", **arrays)
                episodes.append(summary)
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        ok = sum(e["success"] for e in episodes)
        ci = wilson_interval(ok, len(episodes))
        info = {
            "run_id": args.run.name, "policy": args.name or args.rec.parent.name,
            "policy_name": args.name or args.rec.parent.name, "env": "isaac",
            "variant": args.rec.name.removesuffix(".rec"),
            "warmup_steps": 10, "actuators": "PhysX (Isaac)",
            "source": str(args.rec), "isaac_argv": meta.get("argv"),
            "n_episodes": len(episodes), "seed": SEED_BASE, "successes": ok,
            "success_rate": ok / len(episodes), "wilson_95": [ci.low, ci.high],
        }
        (args.run / "run.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        from evalkit.report import write_report

        write_report(args.run)
        print(f"{ok}/{len(episodes)} sucessos; run em {args.run}")


if __name__ == "__main__":
    main()
