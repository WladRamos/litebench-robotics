"""Re-renderiza o dataset do curso no Vials-Lite: mesmos episódios, imagens e juntas daqui.

Cada episódio roda na sua cena reconstruída (``scripts/reconstruct_scenes.py``)
com as ações gravadas em malha aberta, como o ``--replay`` do runner, e é gravado como um
dataset LeRobot v3 com as mesmas features, o mesmo texto da tarefa e o mesmo metadado de fps do
original. Serve para treinar uma política com as imagens daqui (o ACT-Lite da validação cruzada).

Convenção de frames, a do gravador do workshop (ver ``replay_dynamic.py``): ele chama
``env.step(action[t])`` e depois lê a observação ``t``. Aqui, o frame 0 é a observação do reset
(o braço em ``state[0]`` da cena) com ``action[0]``; o frame ``t`` é a observação depois de
``step(action[t])``, com ``action[t]``. O episódio roda até o fim gravado, sem parar no sucesso.
``observation.state`` são as juntas do Vials-Lite, coerentes com as imagens daqui, e não as do
Isaac.

Grave todos os episódios das cenas: assim cada um fica com o mesmo índice do original, e quais
entram no treino se escolhe no ``lerobot-train`` (``--dataset.episodes``), igual para os dois
datasets. ``--episodes`` e ``--max-frames`` são para teste: o LeRobot renumera os episódios a partir
de 0. Ao lado da pasta do dataset fica ``<pasta>_render.jsonl``: por episódio, o índice no original,
se o replay teve sucesso e em que frame.

Uso (uma linha; PowerShell e bash; no Ubuntu o render usa ``MUJOCO_GL=egl``):
    uv run python scripts/render_dataset.py --dataset datasets/so101_teleop_vials_rack_left
        --scenes outputs/scenes/scenes.jsonl --out datasets/so101_teleop_vials_rack_left_vials_lite
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CAMERAS = ("ego", "external_D455")
# features que o LeRobot cria sozinho em todo dataset
DEFAULT_FEATURES = ("timestamp", "frame_index", "episode_index", "index", "task_index")


def render_episode(env, actions: np.ndarray, scene: dict | None, seed: int = 0) -> Iterator[dict]:
    """Frames de um episódio re-renderizado: ``{"observation.state", "observation.images.<câmera>",
    "action", "success"}``, um por linha de ``actions`` (a convenção está na docstring do módulo).
    ``success``: o critério oficial já foi cumprido até esse frame."""
    obs, _ = env.reset(seed=seed, options={"scene": scene} if scene is not None else None)
    success = False
    for t, action in enumerate(np.asarray(actions, np.float32)):
        if t > 0:
            obs, _, terminated, _, _ = env.step(action)
            success = success or terminated
        yield {
            "observation.state": np.asarray(obs["agent_pos"], np.float32),
            **{f"observation.images.{cam}": obs["pixels"][cam] for cam in CAMERAS},
            "action": action,
            "success": success,
        }


def dataset_features(info: dict) -> dict:
    """Features do ``info.json`` do dataset original, sem as padrão e sem os metadados de vídeo
    (o ``LeRobotDataset`` refaz os dois)."""
    return {
        key: {k: v for k, v in feat.items() if k in ("dtype", "shape", "names")}
        for key, feat in info["features"].items()
        if key not in DEFAULT_FEATURES
    }


def read_actions(data_dir: Path, episode: int) -> np.ndarray:
    import pyarrow.parquet as pq

    table = pq.read_table(data_dir / "data" / "chunk-000" / f"file-{episode:03d}.parquet", columns=["action"])
    return np.stack(table.column("action").to_numpy(zero_copy_only=False)).astype(np.float32)


def main() -> None:
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")  # antes de importar o mujoco
    import pandas as pd
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from evalkit.runner import is_gpu_renderer
    from vials_lite.env import VialsLiteEnv

    ap = argparse.ArgumentParser(description="Re-renderiza o dataset do curso no Vials-Lite.")
    ap.add_argument("--dataset", type=Path, required=True, help="pasta do dataset do curso (LeRobot v3)")
    ap.add_argument("--scenes", type=Path, required=True, help="scenes.jsonl de reconstruct_scenes.py")
    ap.add_argument("--out", type=Path, required=True, help="pasta do dataset novo (não pode existir)")
    ap.add_argument("--repo-id", default="local/so101_teleop_vials_rack_left_vials_lite")
    ap.add_argument("--episodes", default="", help="só estes episódios do dataset (ex.: 0,1); padrão: todos")
    ap.add_argument("--max-frames", type=int, default=None, help="corta cada episódio (teste rápido)")
    ap.add_argument("--actuators", default="isaac", choices=("isaac", "menagerie"))
    ap.add_argument("--any-renderer", action="store_true", help="roda mesmo com o render fora da NVIDIA")
    args = ap.parse_args()

    info = json.loads((args.dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    tasks = pd.read_parquet(args.dataset / "meta" / "tasks.parquet")
    if len(tasks) != 1:
        ap.error(f"esperava 1 tarefa no dataset, achei {len(tasks)}")
    task = str(tasks.index[0])
    with args.scenes.open(encoding="utf-8") as f:
        scenes = [json.loads(line) for line in f if line.strip()]
    if args.episodes:
        wanted = {int(e) for e in args.episodes.split(",")}
        scenes = [s for s in scenes if s["episode"] in wanted]
    scenes.sort(key=lambda s: s["episode"])

    env = VialsLiteEnv(variant="eval", actuators=args.actuators, task=task)
    gl = env.gl_renderer()
    print(f"render: {gl}")
    if not is_gpu_renderer(gl) and not args.any_renderer:
        env.close()
        ap.error(f"o render não está na NVIDIA ({gl}); confira MUJOCO_GL=egl e o driver")

    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=info["fps"], features=dataset_features(info),
                               root=args.out, robot_type=info.get("robot_type"), use_videos=True,
                               image_writer_threads=4)
    log_path = args.out.parent / f"{args.out.name}_render.jsonl"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            for row in scenes:
                t0 = time.perf_counter()
                actions = read_actions(args.dataset, row["episode"])[: args.max_frames]
                first_success = None
                for t, frame in enumerate(render_episode(env, actions, row["scene"])):
                    if frame.pop("success") and first_success is None:
                        first_success = t
                    ds.add_frame({**frame, "task": task})
                ds.save_episode()
                rec = {"episode": row["episode"], "frames": len(actions),
                       "success": first_success is not None, "first_success_frame": first_success,
                       "wall_s": round(time.perf_counter() - t0, 1)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"ep {row['episode']:3d}: {len(actions)} frames, "
                      f"{'sucesso' if rec['success'] else 'sem sucesso'} ({rec['wall_s']} s)")
    finally:
        ds.finalize()
        env.close()
    print(f"dataset em {args.out}; log por episódio em {log_path}")


if __name__ == "__main__":
    main()
