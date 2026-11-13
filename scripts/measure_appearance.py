"""Cor do fundo (piso, paredes, tapete) no vídeo externo do dataset e no render.

Para calibrar a luz da lightbox: mede, por episódio, a cor média de regiões que só mostram o fundo.
Em ``--frames`` frames do episódio, a mediana temporal tira o robô e os frascos que passam pelo
piso e pelas paredes; no tapete, que é preto e por onde o braço passa quase o tempo todo, vale o
percentil 10 (o que passa por cima é mais claro que ele). As regiões vêm do render de segmentação
da cena com a câmera calibrada (``lightbox_floor`` fora do tapete, ``lightbox_back``,
``lightbox_left``, ``mat``), com o braço na pose de repouso do 1º frame do dataset, erodidas em
``--erode`` pixels para tolerar a variação da câmera entre episódios (±2 cm) e sem uma
faixa de 2 × ``--erode`` pixels em volta do robô, do rack e dos frascos, que mudam de lugar.

Uso (uma linha; bash e PowerShell iguais; no Ubuntu defina ``MUJOCO_GL=egl``):
    uv run --with av --with pyarrow --with scipy python scripts/measure_appearance.py
        datasets/so101_teleop_vials_rack_left --out outputs/appearance/video.jsonl
        [--episodes 0 4 33] [--masks outputs/appearance/masks.png]

Saída: um JSON por episódio com a média RGB (0–255) de cada região (aparada por luminância: 10–90%;
no tapete, a metade mais escura).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

WIDTH, HEIGHT = 160, 120
REGIONS = {"floor": "lightbox_floor", "left": "lightbox_left", "back": "lightbox_back", "mat": "mat"}


MAT_PERCENTILE = 10
# média aparada por luminância: no tapete, só a metade mais escura (o rack real invade a máscara)
TRIM = {"mat": (0.0, 0.5)}
DEFAULT_TRIM = (0.1, 0.9)


def region_means(stat: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> dict[str, list[float]]:
    """Média RGB aparada de cada região; ``stat`` dá a imagem (H, W, 3) da região ou "default"."""
    out = {}
    for k, mk in masks.items():
        if not mk.any():
            continue
        px = stat.get(k, stat["default"])[mk].reshape(-1, 3)
        lum = px.mean(axis=1)
        lo, hi = np.quantile(lum, TRIM.get(k, DEFAULT_TRIM))
        out[k] = px[(lum >= lo) & (lum <= hi)].mean(axis=0).tolist()
    return out


def region_masks(model, data, camera: str, erode: int) -> dict[str, np.ndarray]:
    """Máscaras das regiões de fundo no render de segmentação, longe do robô, do rack e dos frascos."""
    import mujoco
    from scipy.ndimage import binary_dilation, binary_erosion

    r = mujoco.Renderer(model, HEIGHT, WIDTH)
    r.enable_segmentation_rendering()
    r.update_scene(data, camera=camera)
    seg = r.render()
    r.close()
    geom = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] >= 0)
    background = [model.geom(n).id for n in REGIONS.values()]
    near_objects = binary_dilation(geom & ~np.isin(seg[..., 0], background), iterations=2 * erode)
    out = {}
    for key, name in REGIONS.items():
        mk = geom & (seg[..., 0] == model.geom(name).id)
        out[key] = (binary_erosion(mk, iterations=erode) if erode else mk) & ~near_objects
    return out


def frame_stats(video: Path, n_frames: int) -> dict[str, np.ndarray]:
    """Mediana temporal ("default") e percentil ``MAT_PERCENTILE`` ("mat") de frames 160×120."""
    import av
    from PIL import Image

    with av.open(str(video)) as c:
        images = [f.to_image() for f in c.decode(c.streams.video[0])]
    idx = np.linspace(0, len(images) - 1, n_frames).round().astype(int)
    frames = np.stack([np.asarray(images[i].resize((WIDTH, HEIGHT), Image.BOX)) for i in idx])
    return {"default": np.median(frames, axis=0), "mat": np.percentile(frames, MAT_PERCENTILE, axis=0)}


def rest_pose_mjcf(data_dir: Path) -> np.ndarray:
    """Juntas do 1º frame do episódio 0 (robô em repouso), em radianos do MJCF."""
    import pyarrow.parquet as pq

    from vials_lite.joint_mapping import lerobot_to_radians, usd_radians_to_mjcf

    table = pq.read_table(data_dir / "data" / "chunk-000" / "file-000.parquet", columns=["observation.state"])
    return usd_radians_to_mjcf(lerobot_to_radians(np.array(table.column(0)[0].as_py())))


def main() -> None:
    import mujoco

    from vials_lite.scene import build_model

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data", type=Path, help="pasta do dataset (videos/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="padrão: todos")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--erode", type=int, default=6, help="erosão das máscaras, em pixels de 160×120")
    ap.add_argument("--masks", type=Path, default=None, help="PNG com as máscaras, para conferir")
    args = ap.parse_args()

    model, ids = build_model()
    data = mujoco.MjData(model)
    data.qpos[ids.arm_qpos] = rest_pose_mjcf(args.data)
    mujoco.mj_forward(model, data)
    masks = region_masks(model, data, "external_D455", args.erode)
    print({k: int(v.sum()) for k, v in masks.items()}, "pixels por região")
    if args.masks:
        from PIL import Image

        img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        for color, mk in zip(
            [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)], masks.values(), strict=True
        ):
            img[mk] = color
        args.masks.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).resize((640, 480), Image.NEAREST).save(args.masks)

    folder = args.data / "videos" / "observation.images.external_D455" / "chunk-000"
    videos = sorted(folder.glob("file-*.mp4"))
    if args.episodes is not None:
        videos = [v for v in videos if int(v.stem.split("-")[1]) in set(args.episodes)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for v in videos:
            ep = int(v.stem.split("-")[1])
            means = region_means(frame_stats(v, args.frames), masks)
            f.write(json.dumps({"episode": ep, **means}) + "\n")
            print(ep, {k: np.round(x).astype(int).tolist() for k, x in means.items()}, flush=True)


if __name__ == "__main__":
    main()
