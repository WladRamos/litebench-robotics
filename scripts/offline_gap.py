"""Teste offline do gap visual de uma política.

A política recebe o estado gravado ``state[t]`` de um episódio do dataset do curso com imagens de
fontes diferentes, e o chunk previsto é comparado com as ações gravadas ``action[t : t + 100]``
(erro absoluto médio em unidades do LeRobot, por horizonte). Condições:

* ``isaac``: os frames do próprio dataset (o melhor caso);
* ``trocada``: frames de outro episódio no mesmo t (o que a política faz com a cena errada);
* ``sim``: render do Vials-Lite da cena reconstruída do episódio (``reconstruct_scenes.py``), com o
  braço em ``state[t]`` e a câmera externa, a luz, a cor do robô e o tapete do episódio;
* ``sim_cam_padrao``: igual, com a câmera externa na pose padrão do ``spec``;
* ``ego_sim+ext_isaac`` e ``ego_isaac+ext_sim``: uma câmera de cada fonte.

``--phase pre_grasp`` (padrão): só frames antes do agarre (``replay_dataset.held_interval``), em
que os objetos estão parados na pose do reset e o render da cena reconstruída vale.

``--phase place``: frames com o frasco na mão. Frascos e rack vêm do replay das
ações gravadas (``--replay-run``, um run de ``evalkit.runner --replay``; só episódios em que o replay
teve sucesso): o step k do replay é o frame k + 1 do dataset (ver ``ReplayPolicy``), e o frasco
agarrado é levado do frame da garra do replay para o da garra em ``state[t]``. Mede também, por
cinemática direta, onde a garra estaria no instante em que a demonstração abre a garra
(``place_mm``: distância no plano entre a posição prevista e a gravada, quando esse instante cabe
no chunk). Sem ``sim_cam_padrao``.

``--regions`` (diagnóstico por região), numa câmera de cada vez, com a outra do
Isaac e as regiões pela segmentação do render: ``<câmera>:+<região>`` é o frame do Isaac com os
pixels da região vindos do render (quanto a região simulada atrapalha); ``<câmera>:-<região>`` é o
render com os pixels da região vindos do Isaac (quanto consertá-la ajudaria). Regiões: ``robo``,
``frasco_mao`` (o frasco agarrado; só em ``place``), ``frascos``, ``rack``, ``tapete``, ``fundo``
(lightbox e o que não é geom).

A focal da câmera de punho no dataset é sorteada (12–15 mm); o render usa a da reconstrução
(``ego_fovy`` da cena).

Uso (uma linha; no Ubuntu defina ``MUJOCO_GL=egl``):
    uv run --with av --with pyarrow --with pillow python scripts/offline_gap.py
        datasets/so101_teleop_vials_rack_left --policy checkpoints/act_vials_50k
        --scenes outputs/scenes/scenes.jsonl --out outputs/offline_gap/act_vials_50k.json
        [--episodes 0 3 6] [--stride 10] [--examples 4]
        [--phase place --replay-run outputs/evalkit/replay_scenes_v3_atrito]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

HORIZONS = (1, 10, 30, 100)
CAMS = ("ego", "external_D455")
OTHER_EPISODE_SHIFT = 37  # condição "trocada": episódio (e + 37) mod N
CONDITIONS = ("isaac", "trocada", "sim", "sim_cam_padrao", "ego_sim+ext_isaac", "ego_isaac+ext_sim")
REGIONS = ("robo", "frasco_mao", "frascos", "rack", "tapete", "fundo")


def dataset_task(data_dir: Path) -> str:
    import pyarrow.parquet as pq

    # LeRobot v3: o texto da tarefa é o índice do DataFrame (coluna task_index)
    return str(pq.read_table(data_dir / "meta" / "tasks.parquet").to_pandas().index[0])


def load_episode(data_dir: Path, ep: int) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq

    t = pq.read_table(data_dir / "data" / "chunk-000" / f"file-{ep:03d}.parquet")
    state = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
    action = np.stack(t.column("action").to_numpy(zero_copy_only=False)).astype(np.float32)
    return state, action


def load_frames(data_dir: Path, ep: int, frames: list[int]) -> dict[str, dict[int, np.ndarray]]:
    """Frames RGB uint8 (480, 640, 3) das duas câmeras."""
    import av

    out: dict[str, dict[int, np.ndarray]] = {}
    want = set(frames)
    for cam in CAMS:
        path = data_dir / "videos" / f"observation.images.{cam}" / "chunk-000" / f"file-{ep:03d}.mp4"
        got: dict[int, np.ndarray] = {}
        with av.open(str(path)) as c:
            for i, fr in enumerate(c.decode(c.streams.video[0])):
                if i in want:
                    got[i] = fr.to_ndarray(format="rgb24")
                if i >= max(want):
                    break
        last = got[max(got)]
        out[cam] = {i: got.get(i, last) for i in frames}  # episódio mais curto: o último frame
    return out


class SceneRenderer:
    """Cena reconstruída de um episódio, com o braço posto em cada ``state[t]`` sem física."""

    def __init__(self) -> None:
        import mujoco

        from vials_lite.env import VialsLiteEnv

        self.mj = mujoco
        self.env = VialsLiteEnv(render_images=False)
        self._seg = None
        m, ids = self.env.model, self.env.ids
        # região de cada geom; o frasco agarrado é separado em ``segment``
        self.geom_region = np.full(m.ngeom, REGIONS.index("fundo"))
        for g in range(m.ngeom):
            body = int(m.geom_bodyid[g])
            if body == ids.rack_body:
                self.geom_region[g] = REGIONS.index("rack")
            elif body in ids.vial_bodies:
                self.geom_region[g] = REGIONS.index("frascos")
            elif g == ids.mat_geom:
                self.geom_region[g] = REGIONS.index("tapete")
            elif body != 0:
                self.geom_region[g] = REGIONS.index("robo")

    def load(self, scene: dict) -> None:
        self.env.reset(seed=0, options={"scene": scene})
        m, c = self.env.model, self.env.ids.external_cam
        self.cam = (m.cam_pos[c].copy(), m.cam_quat[c].copy(), float(m.cam_fovy[c]))

    def set_objects(self, rack_pos, rack_quat, vial_pos, vial_quat) -> None:
        """Poses dos objetos (frame do mundo, quaternions wxyz), como gravadas pelo runner."""
        ids, qpos = self.env.ids, self.env.data.qpos
        qpos[ids.rack_qpos : ids.rack_qpos + 3] = rack_pos
        qpos[ids.rack_qpos + 3 : ids.rack_qpos + 7] = rack_quat
        for a, p, q in zip(ids.vial_qpos, vial_pos, vial_quat, strict=True):
            qpos[a : a + 3], qpos[a + 3 : a + 7] = p, q

    def render(self, state: np.ndarray, default_camera: bool = False) -> dict[str, np.ndarray]:
        from vials_lite.joint_mapping import lerobot_to_radians, usd_radians_to_mjcf

        env, m, c = self.env, self.env.model, self.env.ids.external_cam
        env.data.qpos[env.ids.arm_qpos] = usd_radians_to_mjcf(lerobot_to_radians(state.astype(np.float64)))
        if default_camera:
            d = env._default_cam
            m.cam_pos[c], m.cam_quat[c], m.cam_fovy[c] = d["ext_pos"], d["ext_quat"], d["ext_fovy"]
        else:
            m.cam_pos[c], m.cam_quat[c], m.cam_fovy[c] = self.cam
        self.mj.mj_forward(m, env.data)
        return {cam: env._render_camera(cam) for cam in CAMS}

    def segment(self, held_vial: int | None) -> dict[str, np.ndarray]:
        """Índice da região (em ``REGIONS``) de cada pixel, por câmera, na pose do último ``render``."""
        mj, m, ids = self.mj, self.env.model, self.env.ids
        if self._seg is None:
            self._seg = mj.Renderer(m, *self.env._render_camera("ego").shape[:2])
            self._seg.enable_segmentation_rendering()
        out = {}
        for cam in CAMS:
            self._seg.update_scene(self.env.data, camera=cam)
            seg = self._seg.render()
            is_geom = (seg[..., 1] == int(mj.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] >= 0)
            geom = np.where(is_geom, seg[..., 0], 0)
            region = np.where(is_geom, self.geom_region[geom], REGIONS.index("fundo"))
            if held_vial is not None:
                held = is_geom & (m.geom_bodyid[geom] == ids.vial_bodies[held_vial])
                region[held] = REGIONS.index("frasco_mao")
            out[cam] = region
        return out


class Kinematics:
    """Cinemática direta do braço (dados próprios, sem mexer no render): pose do corpo da garra e
    posição do site ``gripperframe`` para juntas em unidades do LeRobot."""

    def __init__(self, model, ids) -> None:
        import mujoco

        self.mj, self.m, self.ids = mujoco, model, ids
        self.d = mujoco.MjData(model)
        self.body = int(model.site_bodyid[ids.gripper_site])

    def __call__(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from vials_lite.joint_mapping import lerobot_to_radians, usd_radians_to_mjcf

        self.d.qpos[self.ids.arm_qpos] = usd_radians_to_mjcf(lerobot_to_radians(joints.astype(np.float64)))
        self.mj.mj_kinematics(self.m, self.d)
        return (self.d.xpos[self.body].copy(), self.d.xquat[self.body].copy(),
                self.d.site_xpos[self.ids.gripper_site].copy())


def carry_vial(kin: Kinematics, joints_from: np.ndarray, joints_to: np.ndarray, pos: np.ndarray,
               quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pose de um frasco preso à garra quando o braço vai de ``joints_from`` para ``joints_to``."""
    from vials_lite.success import quat_inverse, quat_rotate

    mj = kin.mj
    p0, q0, _ = kin(joints_from)
    p1, q1, _ = kin(joints_to)
    inv0 = quat_inverse(q0)
    rel_q = np.zeros(4)
    mj.mju_mulQuat(rel_q, inv0, quat)
    new_q = np.zeros(4)
    mj.mju_mulQuat(new_q, q1, rel_q)
    return p1 + quat_rotate(q1, quat_rotate(inv0, pos - p0)), new_q


def chunk_error(pred: np.ndarray, ref: np.ndarray) -> list[float]:
    """MAE por horizonte (NaN quando o episódio ou o chunk, 50 no pi0.5, acaba antes do horizonte)."""
    n = min(len(pred), len(ref))
    err = np.abs(pred[:n] - ref[:n])
    return [float(err[:h].mean()) if n >= h else float("nan") for h in HORIZONS]


def load_replay(run_dir: Path) -> dict[int, dict]:
    """Episódios do dataset em que o replay teve sucesso: arrays por step do runner."""
    with (run_dir / "episodes.jsonl").open(encoding="utf-8") as f:
        eps = [json.loads(line) for line in f if line.strip()]
    return {e["dataset_episode"]: dict(np.load(run_dir / "steps" / f"ep_{e['episode']:03d}.npz"))
            for e in eps if e["success"]}


def main() -> None:
    from replay_dataset import held_interval

    from evalkit.lerobot_policy import LeRobotPolicy

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data", type=Path, help="pasta do dataset (data/, videos/, meta/)")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--scenes", type=Path, default=Path("outputs/scenes/scenes.jsonl"))
    ap.add_argument("--out", type=Path, required=True, help="resumo em JSON")
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="padrão: todas as cenas")
    ap.add_argument("--stride", type=int, default=10, help="um frame a cada N")
    ap.add_argument("--examples", type=int, default=0, help="salva Isaac × sim dos N primeiros episódios")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--phase", choices=("pre_grasp", "place"), default="pre_grasp")
    ap.add_argument("--replay-run", type=Path, default=Path("outputs/evalkit/replay_scenes_v3_atrito"),
                    help="--phase place: run do evalkit.runner --replay com as poses dos objetos")
    ap.add_argument("--regions", action="store_true", help="acrescenta o diagnóstico por região")
    args = ap.parse_args()

    place = args.phase == "place"
    conditions = [c for c in CONDITIONS if not (place and c == "sim_cam_padrao")]
    if args.regions:
        conditions += [f"{cam}:{sign}{reg}" for sign in "+-" for cam in CAMS for reg in REGIONS
                       if place or reg != "frasco_mao"]
    with args.scenes.open(encoding="utf-8") as f:
        scenes = {r["episode"]: r for r in map(json.loads, f)}
    replay = load_replay(args.replay_run) if place else {}
    episodes = sorted(scenes) if args.episodes is None else args.episodes
    if place:
        episodes = [e for e in episodes if e in replay]
    all_eps = sorted(scenes)
    task = dataset_task(args.data)
    policy = LeRobotPolicy(args.policy, device=args.device)
    sim = SceneRenderer()
    kin = Kinematics(sim.env.model, sim.env.ids)
    errs: dict[str, list[list[float]]] = {c: [] for c in conditions}
    place_mm: dict[str, list[float]] = {c: [] for c in conditions}
    per_episode = []
    t0 = time.time()
    for n, ep in enumerate(episodes):
        state, action = load_episode(args.data, ep)
        held = held_interval(state[:, 5], action[:, 5])
        if place:
            if not held:
                continue
            times = list(range(held[0] + 1, held[1], args.stride))
        else:
            times = list(range(0, (held[0] if held else len(state)) - 5, args.stride))
        if not times:
            continue
        other = all_eps[(all_eps.index(ep) + OTHER_EPISODE_SHIFT) % len(all_eps)]
        isaac = load_frames(args.data, ep, times)
        swapped = load_frames(args.data, other, times)
        sim.load(scenes[ep]["scene"])
        ep_errs: dict[str, list[list[float]]] = {c: [] for c in conditions}
        ep_place: dict[str, list[float]] = {c: [] for c in conditions}
        for t in times:
            v = None
            if place:
                z, k = replay[ep], min(t - 1, len(replay[ep]["vial_pos"]) - 1)
                vial_pos, vial_quat = z["vial_pos"][k].copy(), z["vial_quat"][k].copy()
                v = int(np.argmin(np.linalg.norm(vial_pos - z["gripper_pos"][k], axis=1)))
                vial_pos[v], vial_quat[v] = carry_vial(kin, z["agent_pos"][k], state[t], vial_pos[v],
                                                       vial_quat[v])
                sim.set_objects(z["rack_pos"][k], z["rack_quat"][k], vial_pos, vial_quat)
            real = {c: isaac[c][t] for c in CAMS}
            rendered = sim.render(state[t])
            images = {
                "isaac": real,
                "trocada": {c: swapped[c][t] for c in CAMS},
                "sim": rendered,
                "ego_sim+ext_isaac": {"ego": rendered["ego"], "external_D455": real["external_D455"]},
                "ego_isaac+ext_sim": {"ego": real["ego"], "external_D455": rendered["external_D455"]},
            }
            if args.regions:
                seg = sim.segment(v)
                for cam in CAMS:
                    for i, reg in enumerate(REGIONS):
                        mask = (seg[cam] == i)[..., None]
                        images[f"{cam}:+{reg}"] = {**real, cam: np.where(mask, rendered[cam], real[cam])}
                        images[f"{cam}:-{reg}"] = {**real, cam: np.where(mask, real[cam], rendered[cam])}
            if not place:
                default_cam = sim.render(state[t], default_camera=True)
                images["sim_cam_padrao"] = {"ego": rendered["ego"],
                                            "external_D455": default_cam["external_D455"]}
            ref = action[t : t + max(HORIZONS)]
            r = held[1] - t if place else None  # índice, no chunk, de quando a demonstração abre a garra
            for c in conditions:
                pred = policy.predict_chunk({"agent_pos": state[t], "pixels": images[c]}, task)
                ep_errs[c].append(chunk_error(pred, ref))
                if place and r < min(len(pred), len(ref)):
                    ep_place[c].append(1000 * float(np.linalg.norm((kin(pred[r])[2] - kin(ref[r])[2])[:2])))
            if n < args.examples and t == times[len(times) // 2]:
                from PIL import Image

                row = lambda ims: np.concatenate([ims[c] for c in CAMS], 1)  # noqa: E731
                folder = args.out.with_suffix("")
                folder.mkdir(parents=True, exist_ok=True)
                grid = np.concatenate([row(real), row(rendered)], 0)
                Image.fromarray(grid).save(folder / f"ep{ep:03d}_t{t}.png")
        for c in conditions:
            errs[c] += ep_errs[c]
            place_mm[c] += ep_place[c]
        per_episode.append({"episode": ep, "frames": len(times),
                            **{c: np.nanmean(ep_errs[c], 0).round(3).tolist() for c in conditions},
                            **({f"place_mm_{c}": ep_place[c] for c in conditions} if place else {})})
        last = per_episode[-1]
        print(f"[{n + 1}/{len(episodes)} {time.time() - t0:4.0f}s] ep {ep:3d}: {len(times)} frames;"
              f" h={HORIZONS[2]}: isaac {last['isaac'][2]:.1f}, sim {last['sim'][2]:.1f},"
              f" trocada {last['trocada'][2]:.1f}", flush=True)  # h = 30 cabe no chunk do pi0.5

    summary = {
        "policy": args.policy, "scenes": str(args.scenes), "task": task, "stride": args.stride,
        "phase": args.phase, "horizons": HORIZONS, "n_frames": len(errs["isaac"]),
        "n_episodes": len(per_episode),
        "mae": {c: np.nanmean(errs[c], 0).round(3).tolist() for c in conditions},
        "per_episode": per_episode,
    }
    if place:
        summary["replay_run"] = str(args.replay_run)
        summary["place_mm_median"] = {c: round(float(np.median(place_mm[c])), 1) for c in conditions}
        summary["place_mm_p90"] = {c: round(float(np.percentile(place_mm[c], 90)), 1) for c in conditions}
        summary["place_n"] = len(place_mm["isaac"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    where = "com o frasco na mão" if place else "antes do agarre"
    print(f"\n{summary['n_episodes']} episódios, {summary['n_frames']} frames {where};"
          f" MAE por horizonte {HORIZONS} (unidades do LeRobot)")
    for c in conditions:
        extra = f"   garra na soltura: {summary['place_mm_median'][c]:5.1f} mm (mediana)" if place else ""
        print(f"  {c:20s}", "  ".join(f"{v:5.2f}" for v in summary["mae"][c]), extra)
    if place:
        print(f"  ({summary['place_n']} frames em que o instante da soltura cabe no chunk)")


if __name__ == "__main__":
    main()
