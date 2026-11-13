"""Calibra as constantes de luz (``spec.LightingCalibration``) contra o vídeo do dataset.

O dataset foi gravado na variante DR: a cada episódio, exposição da lightbox em −3..1, do domo em
−4..3 e temperatura do domo em 2500–9500 K. O render do Isaac não se traduz em intensidades do
MuJoCo, então este script sorteia ``--samples`` iluminações com essas mesmas faixas, renderiza a
cena (160×120, câmera externa, braço em repouso) e ajusta as constantes para que os quantis da cor
do piso e da parede do fundo, medidas como em ``measure_appearance.py``, batam com os do vídeo.
Como o dataset é todo DR, um termo extra prende a variante ``eval`` (só a fita, exposição 0): o
piso dela deve ficar no brilho mediano do vídeo (premissa, ver ``spec.LightingCalibration``).

As cores do vídeo vêm das cenas reconstruídas (``reconstruct_scenes.py``, ``fit.appearance_rgb``),
medidas com o tapete e a câmera de cada episódio. As de ``measure_appearance.py`` (tapete na pose
padrão) misturavam tapete e piso nos episódios com o tapete muito girado; o script ainda
aceita esse formato.

Uso (uma linha; no Ubuntu defina ``MUJOCO_GL=egl``):
    uv run --with av --with pyarrow --with scipy python scripts/fit_lighting.py
        datasets/so101_teleop_vials_rack_left outputs/scenes/scenes.jsonl
        [--samples 120] [--maxfev 400]

Imprime as constantes ajustadas e os quantis antes e depois; os valores vão à mão para o spec.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

QUANTILES = np.linspace(0.05, 0.95, 10)
# Os dados não separam luz direcional de ambiente na fita (com tudo livre, o ajuste zera a
# direcional e a imagem fica chapada, sem sombra). Fração direcional da fita fixa, escolhida a
# olho contra o vídeo, que tem sombreado e sombras suaves; a do domo é livre (é ela que deixa a
# parede do fundo, de frente para a abertura, mais clara que o piso, como no vídeo).
DIFFUSE_FRACTION = 0.4
PARAMS = (
    "rect_total",
    "dome_total",
    "dome_fraction",
    "exposure_scale",
    "temperature_strength",
    "white_k",
    "dome_elevation_deg",
)
SCALE = {"white_k": 1000.0, "dome_elevation_deg": 100.0}  # o otimizador vê tudo em ordem 1


def to_calib(z: np.ndarray):
    from vials_lite import spec

    v = {p: abs(x) * SCALE.get(p, 1.0) for p, x in zip(PARAMS, z, strict=True)}
    f, fd = DIFFUSE_FRACTION, min(v["dome_fraction"], 1.0)
    return spec.LightingCalibration(
        rect_diffuse=f * v["rect_total"],
        rect_ambient=(1 - f) * v["rect_total"],
        dome_diffuse=fd * v["dome_total"],
        dome_ambient=(1 - fd) * v["dome_total"],
        exposure_scale=v["exposure_scale"],
        temperature_strength=v["temperature_strength"],
        white_k=v["white_k"],
        dome_elevation_deg=min(v["dome_elevation_deg"], 89.0),
    )


def from_calib(c) -> np.ndarray:
    return np.array(
        [
            c.rect_diffuse + c.rect_ambient,
            c.dome_diffuse + c.dome_ambient,
            c.dome_diffuse / max(c.dome_diffuse + c.dome_ambient, 1e-6),
            c.exposure_scale,
            c.temperature_strength,
            c.white_k / SCALE["white_k"],
            c.dome_elevation_deg / SCALE["dome_elevation_deg"],
        ]
    )


def features(rows: list[dict]) -> dict[str, np.ndarray]:
    """Luminância do piso e do fundo e razão R/B do piso, por episódio (ou por amostra)."""
    floor, back = np.array([r["floor"] for r in rows]), np.array([r["back"] for r in rows])
    return {
        "floor": floor.mean(axis=1),
        "back": back.mean(axis=1),
        "floor_rb": floor[:, 0] / np.maximum(floor[:, 2], 1.0),
    }


def quantile_loss(sim: dict[str, np.ndarray], real: dict[str, np.ndarray]) -> float:
    """Distância quadrática média entre quantis, em unidades de 50 (luminância) e de 0,1 (R/B)."""
    total = 0.0
    for k, scale in (("floor", 50.0), ("back", 50.0), ("floor_rb", 0.1)):
        d = np.quantile(sim[k], QUANTILES) - np.quantile(real[k], QUANTILES)
        total += float(np.mean((d / scale) ** 2))
    return total


def main() -> None:
    import mujoco
    from measure_appearance import HEIGHT, WIDTH, region_masks, region_means, rest_pose_mjcf
    from scipy.optimize import minimize

    from vials_lite import scene, spec

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data", type=Path, help="pasta do dataset (para a pose de repouso)")
    ap.add_argument("video_stats", type=Path,
                    help="cenas de reconstruct_scenes.py (ou saída de measure_appearance.py)")
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--maxfev", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with args.video_stats.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    # cenas reconstruídas: as cores vêm das máscaras da cena do episódio (tapete e câmera no lugar)
    real = features([r["fit"]["appearance_rgb"] if "fit" in r else r for r in rows])

    model, ids = scene.build_model()
    data = mujoco.MjData(model)
    data.qpos[ids.arm_qpos] = rest_pose_mjcf(args.data)
    mujoco.mj_forward(model, data)
    masks = region_masks(model, data, "external_D455", 6)
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)

    rng = np.random.default_rng(args.seed)
    draws = np.column_stack(
        [
            rng.uniform(*spec.RESET.lightbox_exposure, args.samples),
            rng.uniform(*spec.DR.sky_exposure, args.samples),
            rng.uniform(*spec.DR.sky_temperature_k, args.samples),
        ]
    )

    def aim_dome(calib: spec.LightingCalibration) -> None:
        model.light_dir[ids.dome_light] = scene.dome_direction(calib.dome_elevation_deg)
        mujoco.mj_camlight(model, data)  # light_xdir vem do mj_camlight, não do mj_kinematics

    def simulate(calib: spec.LightingCalibration) -> dict[str, np.ndarray]:
        aim_dome(calib)
        rows = []
        for e_rect, e_dome, temp in draws:
            c = scene.light_colors(e_rect, e_dome, temp, calib)
            n = len(ids.rect_lights)
            model.light_diffuse[ids.rect_lights] = c["rect"] / n
            model.light_diffuse[ids.dome_light] = c["dome"]
            model.light_ambient[ids.fill_light] = c["ambient"]
            renderer.update_scene(data, camera="external_D455")
            rows.append(region_means({"default": renderer.render().astype(float)}, masks))
        return features(rows)

    def eval_floor(calib: spec.LightingCalibration) -> float:
        """Brilho do piso na variante eval (só a fita, exposição 0)."""
        c = scene.light_colors(0.0, None, 6500.0, calib)
        model.light_diffuse[ids.rect_lights] = c["rect"] / len(ids.rect_lights)
        model.light_diffuse[ids.dome_light] = c["dome"]
        model.light_ambient[ids.fill_light] = c["ambient"]
        renderer.update_scene(data, camera="external_D455")
        return features([region_means({"default": renderer.render().astype(float)}, masks)])["floor"][0]

    def loss(z: np.ndarray) -> float:
        calib = to_calib(z)
        prior = ((eval_floor(calib) - np.median(real["floor"])) / 50.0) ** 2
        return quantile_loss(simulate(calib), real) + prior

    def report(label: str, calib: spec.LightingCalibration) -> None:
        sim = simulate(calib)
        print(
            f"{label}: perda {quantile_loss(sim, real):.4f}; piso na eval (exposição 0)"
            f" {eval_floor(calib):.0f}, mediana do vídeo {np.median(real['floor']):.0f}"
        )
        for k in sim:
            print(
                f"  {k:9s} vídeo p5/50/95 {np.quantile(real[k], [0.05, 0.5, 0.95]).round(2)}"
                f"  sim {np.quantile(sim[k], [0.05, 0.5, 0.95]).round(2)}"
            )

    z0 = from_calib(spec.LIGHTING)
    report("atual", spec.LIGHTING)
    names = np.array(PARAMS)
    starts = [
        z0,
        z0 * np.where(names == "dome_total", 3.0, 1.0) * np.where(names == "dome_fraction", 2.0, 1.0),
    ]
    runs = [
        minimize(loss, z, method="Powell", options={"maxfev": args.maxfev, "xtol": 1e-3, "ftol": 1e-5})
        for z in starts
    ]
    res = min(runs, key=lambda r: r.fun)
    print("perdas por partida:", [round(float(r.fun), 4) for r in runs])
    best = to_calib(res.x)
    report(f"ajustado ({sum(r.nfev for r in runs)} avaliações)", best)
    print({k: round(v, 4) for k, v in dataclasses.asdict(best).items()})
    renderer.close()


if __name__ == "__main__":
    main()
