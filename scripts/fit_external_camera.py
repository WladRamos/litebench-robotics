"""Ajuste da câmera externa ao vídeo de cada episódio do dataset do curso.

A pose e o FOV da câmera extraídos dos USD não reproduzem o vídeo, e o dataset foi gravado na
variante DR: a cor do robô e a luz mudam por episódio, e a câmera varia muito além da randomização
do código atual. Este script estima, por episódio, o deslocamento da câmera em relação à pose do
``spec`` e o FOV vertical, para obter a distribuição real.

Método, independente da cor do robô:

* observado: máscara do que muda ao longo do episódio, |frame − mediana temporal| > limiar, em
  ``--frames`` frames igualmente espaçados, em 160×120;
* simulado: silhueta do robô (render de segmentação) com o braço em ``observation.state`` dos mesmos
  frames (frame i do vídeo ↔ ``state[i]``); a mudança prevista é ``silhueta XOR silhueta mediana``
  (onde o robô está agora, contra onde está na maior parte do tempo);
* objetivo: 1 − IoU entre as duas máscaras, minimizado por Powell em 7 parâmetros (deslocamento de
  posição no mundo, rotação euler xyz no frame da câmera, FOV vertical), com várias partidas.

Sombras, frascos carregados e o robô parado contaminam a máscara observada: a IoU absoluta fica
baixa (~0,5) mesmo quando o ajuste está certo. Compare episódios entre si e confira as sobreposições
(``--overlays``). FOV e distância se compensam em parte; ``--fovy`` fixa o FOV para um segundo
passo só com a pose.

Uso (uma linha; bash e PowerShell iguais; no Ubuntu defina ``MUJOCO_GL=egl``):
    uv run --with av --with pyarrow --with scipy --with pillow python scripts/fit_external_camera.py
        datasets/so101_teleop_vials_rack_left --out outputs/camera_fit/fit.jsonl
        [--episodes 0 4 33] [--workers 12] [--fovy 52] [--overlays outputs/camera_fit/overlays]

A pasta do dataset precisa de ``data/chunk-000/file-NNN.parquet`` e
``videos/observation.images.external_D455/chunk-000/file-NNN.mp4`` (um arquivo por episódio).
O resultado é um JSON por linha e por episódio, gravado assim que o episódio termina.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

WIDTH, HEIGHT = 160, 120
CHANGE_THRESHOLD = 25  # diferença máxima entre canais RGB (0–255) para contar como mudança
VIDEO_KEY = "observation.images.external_D455"


def change_mask(frames: np.ndarray, threshold: float = CHANGE_THRESHOLD) -> np.ndarray:
    """Máscara (N, H, W) do que difere da mediana temporal. frames: (N, H, W, 3)."""
    f = frames.astype(np.int16)
    return np.abs(f - np.median(f, axis=0)).max(axis=-1) > threshold


def predicted_change(silhouettes: np.ndarray) -> np.ndarray:
    """Mudança que as silhuetas (N, H, W) produziriam contra a mediana temporal delas."""
    return silhouettes ^ (silhouettes.mean(axis=0) > 0.5)[None]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def load_episode(data_dir: Path, episode: int, n_frames: int):
    """(índices dos frames, frames 160×120 RGB, observation.state desses frames)."""
    import av
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(
        data_dir / "data" / "chunk-000" / f"file-{episode:03d}.parquet", columns=["observation.state"]
    )
    state = np.array(table.column("observation.state").to_pylist(), dtype=np.float64)
    video = data_dir / "videos" / VIDEO_KEY / "chunk-000" / f"file-{episode:03d}.mp4"
    with av.open(str(video)) as container:
        images = [f.to_image() for f in container.decode(container.streams.video[0])]
    if len(images) != len(state):
        raise ValueError(f"episódio {episode}: {len(images)} frames no vídeo, {len(state)} no parquet")
    idx = np.linspace(0, len(state) - 1, n_frames).round().astype(int)
    frames = np.stack([np.asarray(images[i].resize((WIDTH, HEIGHT), Image.BOX)) for i in idx])
    return idx, frames, state[idx], images


class SilhouetteRenderer:
    """Renderiza a silhueta do robô na câmera externa para um vetor de parâmetros x (7,)."""

    def __init__(self):
        import mujoco

        from vials_lite.scene import build_model

        self._mj = mujoco
        self.model, self.ids = build_model()
        self.data = mujoco.MjData(self.model)
        m, cam = self.model, self.ids.external_cam
        self.default_pos, self.default_quat = m.cam_pos[cam].copy(), m.cam_quat[cam].copy()
        base = m.body("base").id

        def in_robot(b: int) -> bool:
            while b != 0:
                if b == base:
                    return True
                b = m.body_parentid[b]
            return False

        self.robot_geom = np.array([in_robot(m.geom_bodyid[g]) for g in range(m.ngeom)])
        # int: comparar o array com o enum pybind cai em comparação elemento a elemento (~10× mais lenta)
        self._geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
        self.renderer = mujoco.Renderer(m, HEIGHT, WIDTH)
        self.renderer.enable_segmentation_rendering()

    def set_camera(self, x: np.ndarray) -> None:
        from vials_lite.env import _quat_mul, quat_from_euler_xyz

        m, cam = self.model, self.ids.external_cam
        m.cam_pos[cam] = self.default_pos + x[:3]
        m.cam_quat[cam] = _quat_mul(self.default_quat, quat_from_euler_xyz(*x[3:6]))
        m.cam_fovy[cam] = x[6]

    def silhouettes(self, x: np.ndarray, qpos: list[np.ndarray]) -> np.ndarray:
        mj, m, d = self._mj, self.model, self.data
        self.set_camera(x)
        out = np.empty((len(qpos), HEIGHT, WIDTH), dtype=bool)
        for k, q in enumerate(qpos):
            d.qpos[self.ids.arm_qpos] = q
            mj.mj_kinematics(m, d)
            mj.mj_camlight(m, d)
            self.renderer.update_scene(d, camera=self.ids.external_cam)
            seg = self.renderer.render()
            oid, otype = seg[..., 0], seg[..., 1]
            hit = (otype == self._geom_type) & (oid >= 0)
            out[k] = hit & self.robot_geom[np.clip(oid, 0, m.ngeom - 1)]
        return out

    def close(self) -> None:
        self.renderer.close()


_worker: SilhouetteRenderer | None = None


def fit_episode(job: dict) -> dict:
    from scipy.optimize import minimize

    from vials_lite.joint_mapping import lerobot_to_radians, usd_radians_to_mjcf

    global _worker
    if _worker is None:
        _worker = SilhouetteRenderer()
    t0 = time.time()
    ep = job["episode"]
    idx, frames, state, images = load_episode(Path(job["data"]), ep, job["frames"])
    observed = change_mask(frames)
    qpos = [usd_radians_to_mjcf(lerobot_to_radians(s)) for s in state]
    fixed_fovy = job["fovy"]

    def unpack(z: np.ndarray) -> np.ndarray:
        return np.append(z, fixed_fovy) if fixed_fovy is not None else z

    def loss(z: np.ndarray) -> float:
        return 1.0 - iou(predicted_change(_worker.silhouettes(unpack(z), qpos)), observed)

    starts = [fixed_fovy] if fixed_fovy is not None else job["starts"]
    runs = []
    for fovy in starts:
        z0 = np.zeros(6) if fixed_fovy is not None else np.array([0, 0, 0, 0, 0, 0, fovy], float)
        res = minimize(loss, z0, method="Powell", options={"maxfev": 1500, "xtol": 1e-3, "ftol": 1e-4})
        runs.append({"start_fovy": fovy, "iou": 1.0 - float(res.fun), "x": unpack(res.x).tolist()})
    best = max(runs, key=lambda r: r["iou"])
    x = np.array(best["x"])
    # antes do set_camera(x): o loss da pose padrão move a câmera; se viesse depois,
    # cam_pos/cam_quat_wxyz sairiam iguais à pose padrão em todos os episódios
    iou_default = 1.0 - loss(np.zeros(6) if fixed_fovy is not None else np.array([0] * 6 + [74.3]))
    _worker.set_camera(x)
    m, cam = _worker.model, _worker.ids.external_cam
    result = {
        "episode": ep,
        "iou": best["iou"],
        "iou_default": iou_default,
        "base_pos": _worker.default_pos.tolist(),
        "base_quat_wxyz": _worker.default_quat.tolist(),
        "fovy": float(x[6]),
        "dpos": x[:3].tolist(),
        "drot_xyz": x[3:6].tolist(),
        "cam_pos": m.cam_pos[cam].tolist(),
        "cam_quat_wxyz": m.cam_quat[cam].tolist(),
        "observed_fraction": float(observed.mean()),
        "runs": runs,
        "seconds": round(time.time() - t0, 1),
    }
    if job["overlays"]:
        _save_overlay(Path(job["overlays"]), ep, x, images, idx, qpos, state)
    return result


def _save_overlay(folder: Path, ep: int, x: np.ndarray, images, idx, qpos, state) -> None:
    """Frame do meio do episódio com a silhueta ajustada em vermelho, em 640×480."""
    from PIL import Image

    k = len(idx) // 2
    sil = _worker.silhouettes(x, [qpos[k]])[0]
    img = np.asarray(images[idx[k]].resize((WIDTH, HEIGHT))).copy()
    img[sil] = (0.5 * img[sil] + [127, 0, 0]).astype(np.uint8)
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).resize((640, 480), Image.NEAREST).save(folder / f"ep{ep:03d}.png")


def main() -> None:
    import multiprocessing as mp

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data", type=Path, help="pasta do dataset (data/ e videos/)")
    ap.add_argument("--out", type=Path, required=True, help="arquivo .jsonl de saída (acrescenta)")
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="padrão: todos")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--frames", type=int, default=16, help="frames por episódio")
    ap.add_argument("--starts", type=float, nargs="+", default=[45.0, 60.0], help="FOVs iniciais")
    ap.add_argument("--fovy", type=float, default=None, help="fixa o FOV e ajusta só a pose")
    ap.add_argument("--overlays", type=Path, default=None, help="pasta para as sobreposições")
    args = ap.parse_args()

    episodes = args.episodes
    if episodes is None:
        files = sorted((args.data / "data" / "chunk-000").glob("file-*.parquet"))
        episodes = [int(p.stem.split("-")[1]) for p in files]
    done = set()
    if args.out.is_file():
        with args.out.open(encoding="utf-8") as f:
            done = {json.loads(line)["episode"] for line in f if line.strip()}
    todo = [e for e in episodes if e not in done]
    print(f"{len(todo)} episódios a ajustar ({len(done)} já em {args.out}), {args.workers} processos")
    jobs = [
        {
            "data": str(args.data),
            "episode": e,
            "frames": args.frames,
            "starts": args.starts,
            "fovy": args.fovy,
            "overlays": str(args.overlays) if args.overlays else None,
        }
        for e in todo
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool, args.out.open("a", encoding="utf-8") as f:
        for n, r in enumerate(pool.imap_unordered(fit_episode, jobs), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(
                f"[{n}/{len(jobs)} {time.time() - t0:5.0f}s] ep {r['episode']:3d}: IoU {r['iou']:.2f}"
                f" (padrão {r['iou_default']:.2f}), fovy {r['fovy']:5.1f},"
                f" dpos {np.round(r['dpos'], 3).tolist()}",
                flush=True,
            )


if __name__ == "__main__":
    main()
