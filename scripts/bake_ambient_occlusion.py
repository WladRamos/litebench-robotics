"""Grava a oclusão de ambiente nas malhas visuais do Vials-Lite.

O Isaac renderiza com path tracing: dentro dos furos do rack chega pouca luz e eles ficam escuros.
O MuJoCo não calcula oclusão e mostra os furos claros, com o fundo iluminado; no teste offline, o
rack era a região da câmera externa que mais pesava no erro do ACT. Este script calcula, para cada
canto (vértice, normal) da malha, a fração do céu visível: ``--rays`` raios em direções sorteadas
(cosseno-ponderadas, semente fixa) no hemisfério da normal, contra a própria malha (``mj_multiRay``).
O valor vai para a coordenada de textura u do OBJ (``scene.ao_to_u``); ``scene.build_model`` aplica
às malhas com ``vt`` uma rampa de cinza (``scene.AO_MATERIAL``) que multiplica a cor do geom.

Só a própria malha oculta: o chão e os outros objetos mudam de lugar a cada episódio. Rodar de
novo depois de ``export_workshop_meshes.py`` (que grava os OBJ sem ``vt``); rodar duas vezes dá o
mesmo arquivo.

Uso (uma linha; bash e PowerShell iguais):
    uv run python scripts/bake_ambient_occlusion.py [--meshes rack] [--rays 256]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SURFACE_OFFSET_M = 2e-4  # o raio sai um pouco acima da superfície para não bater no próprio triângulo
HEADER = "Oclusão de ambiente em vt (u) por scripts/bake_ambient_occlusion.py"


def parse_obj(text: str) -> tuple[list[str], np.ndarray, np.ndarray, list[list[tuple[int, int]]]]:
    """(comentários, vértices, normais, faces como listas de (vértice, normal), índices a partir de 0)."""
    comments, verts, normals, faces = [], [], [], []
    for line in text.splitlines():
        tag, *rest = line.split() or [""]
        if tag == "#" and HEADER not in line:
            comments.append(line)
        elif tag == "v":
            verts.append([float(x) for x in rest[:3]])
        elif tag == "vn":
            normals.append([float(x) for x in rest[:3]])
        elif tag == "f":
            corners = [c.split("/") for c in rest]
            faces.append([(int(c[0]) - 1, int(c[2]) - 1) for c in corners])
    return comments, np.array(verts), np.array(normals), faces


def hemisphere(n_rays: int, seed: int = 0) -> np.ndarray:
    """Direções cosseno-ponderadas no hemisfério +z (n_rays, 3)."""
    u = np.random.default_rng(seed).random((n_rays, 2))
    r, phi = np.sqrt(u[:, 0]), 2.0 * np.pi * u[:, 1]
    return np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1.0 - u[:, 0])], axis=1)


def tangent_frame(n: np.ndarray) -> np.ndarray:
    """Matriz cujas colunas são (t, b, n), base ortonormal com n na terceira coluna."""
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t = np.cross(n, a)
    t /= np.linalg.norm(t)
    return np.stack([t, np.cross(n, t), n], axis=1)


def sky_fraction(path: Path, verts: np.ndarray, normals: np.ndarray, corners: list[tuple[int, int]],
                 n_rays: int) -> dict[tuple[int, int], float]:
    """Fração dos raios de cada canto que não bate na malha."""
    import mujoco

    s = mujoco.MjSpec()
    s.add_mesh(name="mesh", file=str(path.resolve()))
    # o compilador recentra a malha e põe o geom na pose que a devolve ao frame do arquivo
    s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="mesh")
    m = s.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    local = hemisphere(n_rays)
    geomid = np.zeros(n_rays, np.int32)
    dist = np.zeros(n_rays)
    out = {}
    for vi, ni in corners:
        n = normals[ni] / np.linalg.norm(normals[ni])
        dirs = local @ tangent_frame(n).T
        origin = verts[vi] + SURFACE_OFFSET_M * n
        mujoco.mj_multiRay(m, d, origin, dirs.ravel(), None, 1, -1, geomid, dist, None, n_rays, 1.0)
        out[(vi, ni)] = float(np.mean(geomid < 0))
    return out


def main() -> None:
    from vials_lite.scene import ao_to_u, assets_dir

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--meshes", nargs="+", default=["rack"])
    ap.add_argument("--rays", type=int, default=256)
    args = ap.parse_args()

    for mesh in args.meshes:
        path = assets_dir() / f"{mesh}.obj"
        comments, verts, normals, faces = parse_obj(path.read_text(encoding="utf-8"))
        corners = sorted({c for f in faces for c in f})
        sky = sky_fraction(path, verts, normals, corners, args.rays)
        index = {c: k + 1 for k, c in enumerate(corners)}
        lines = [*comments, f"# {HEADER} ({args.rays} raios por canto)."]
        lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
        lines += [f"vn {x:.4f} {y:.4f} {z:.4f}" for x, y, z in normals]
        lines += [f"vt {ao_to_u(sky[c]):.4f} 0.5" for c in corners]
        lines += ["f " + " ".join(f"{v + 1}/{index[(v, n)]}/{n + 1}" for v, n in f) for f in faces]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        values = np.array(list(sky.values()))
        print(f"{mesh}: {len(corners)} cantos; céu visível p5/50/95 ="
              f" {np.percentile(values, [5, 50, 95]).round(2).tolist()}")


if __name__ == "__main__":
    main()
