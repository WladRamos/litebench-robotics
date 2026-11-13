"""Exporta as malhas visuais do rack e do frasco dos USD do workshop para OBJ.

As malhas vão para ``assets/vials_lite/`` (versionadas; Apache-2.0, atribuição no NOTICE) e são só
visuais: a colisão continua sendo a dos primitivos de ``scene.py``. Coordenadas no frame do próprio
asset, em metros (o mesmo de ``spec``: rack com x, y ∈ [0; 0,12]; frasco com origem na base do
corpo e eixo +z ao longo do tubo). Normais por canto, como no USD (arestas vivas do rack), salvo na
serrilha da tampa, onde viram radiais (``smooth_knurl``); a serrilha volta como
textura, pela coordenada u (``knurl_u``). Depois deste script, rode
``bake_ambient_occlusion.py`` (a oclusão de ambiente do rack vai no OBJ).

Uso (precisa do extra ``usd`` e do clone do workshop ao lado deste repositório, assets/README.md):
    uv run python scripts/export_workshop_meshes.py [--workshop ../Sim-to-Real-SO-101-Workshop]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
USD_DIR = Path("source") / "sim_to_real_so101" / "assets" / "usd"
# arquivo de saída: (USD, prims de malha). O colisor do frasco ("/Vial/collider") fica de fora.
MESHES = {
    "rack.obj": ("Vial_rack_simple.usda", ["/World/Body1/Mesh/Mesh"]),
    "vial_body.obj": ("Vial_opaque.usda", ["/Vial/Mesh_002"]),
    "vial_cap.obj": ("Vial_opaque.usda", ["/Vial/Mesh", "/Vial/Mesh_001"]),
}


# Serrilha da tampa (raio externo 16,2–17 mm): a 640 × 480 cada dente tem menos de um pixel. O Isaac
# integra a luz no pixel e mostra a tampa lisa; no MuJoCo os dentes viram listras (moiré), mesmo com
# normais radiais e sem sombras. Os vértices da serrilha vão para um raio só (cilindro liso) e as
# normais laterais ficam radiais. Por dentro (parede interna em 15–16,2 mm) nada muda.
KNURL_MIN_RADIUS_M = 0.0162
KNURL_RADIUS_M = 0.01695  # ponta dos dentes: a silhueta da tampa no vídeo é a envoltória da serrilha
SMOOTH_KNURL = {"vial_cap.obj"}
# A serrilha volta como textura (scene.KNURL_MATERIAL, com mipmap: de longe vira a média, de perto, na
# câmera de punho, aparecem os sulcos). Medido na malha: 100 dentes (período de 3,6°, um
# começando em −180°) na faixa z 86,2–99,2 mm (os sulcos vão até o chanfro do topo); embaixo, uma
# faixa lisa até 85 mm.
KNURL_TEETH = 100
KNURL_Z_M = (0.0862, 0.0992)
KNURL_FLAT_U = 0.25  # fora da faixa: u fixo no meio do dente (textura clara)


def smooth_knurl(verts: np.ndarray, faces: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Serrilha (eixo z, raio ≥ ``KNURL_MIN_RADIUS_M``) como cilindro liso: vértices no raio
    ``KNURL_RADIUS_M`` e normais laterais radiais."""
    r = np.hypot(verts[:, 0], verts[:, 1])
    knurl = r >= KNURL_MIN_RADIUS_M
    v = verts.copy()
    v[knurl, :2] *= (KNURL_RADIUS_M / r[knurl])[:, None]
    p = v[faces]  # (F, 3, 3)
    rc = np.hypot(p[..., 0], p[..., 1])
    side = (np.abs(normals[..., 2]) < 0.5) & knurl[faces]
    n = normals.copy()
    n[side] = np.stack([p[side, 0] / rc[side], p[side, 1] / rc[side], np.zeros(side.sum())], axis=-1)
    return v, n


def knurl_u(verts: np.ndarray, faces: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Coordenada u por canto (F, 3), em períodos de dente a partir de −180°, nas faces laterais da
    faixa serrilhada; nas outras, ``KNURL_FLAT_U``. Ângulos desenrolados por face (a textura repete)."""
    p = verts[faces]
    r = np.hypot(p[..., 0], p[..., 1])
    z_ok = (p[..., 2] >= KNURL_Z_M[0] - 1e-5) & (p[..., 2] <= KNURL_Z_M[1] + 1e-5)
    band = np.all(z_ok & (r >= KNURL_MIN_RADIUS_M) & (np.abs(normals[..., 2]) < 0.5), axis=1)
    theta = np.arctan2(p[..., 1], p[..., 0])
    theta = theta[:, :1] + (theta - theta[:, :1] + np.pi) % (2 * np.pi) - np.pi
    u = (theta + np.pi) * KNURL_TEETH / (2 * np.pi)
    return np.where(band[:, None], u, KNURL_FLAT_U)


def read_mesh(stage, path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(vértices (V, 3), triângulos (F, 3), normais por canto (F, 3, 3)) no frame do asset."""
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(path)
    mesh = UsdGeom.Mesh(prim)
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    if not np.all(counts == 3):
        raise ValueError(f"{path}: só malhas trianguladas (contagens {set(counts.tolist())})")
    xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    points = np.array(mesh.GetPointsAttr().Get(), dtype=float)
    verts = points @ xf[:3, :3] + xf[3, :3]  # matrizes do USD são row-major (p · M)
    faces = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    normals = np.array(mesh.GetNormalsAttr().Get(), dtype=float)
    interp = mesh.GetNormalsInterpolation()
    if interp == UsdGeom.Tokens.faceVarying:
        normals = normals.reshape(-1, 3, 3)
    elif interp == UsdGeom.Tokens.vertex:
        normals = normals[faces]
    else:
        raise ValueError(f"{path}: normais {interp} não suportadas")
    rot = xf[:3, :3]
    normals = normals @ np.linalg.inv(rot).T if not np.allclose(rot, np.eye(3)) else normals
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
    return verts, faces, normals


def write_obj(
    path: Path,
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    header: str,
    u: list[np.ndarray] | None = None,
) -> None:
    """OBJ com normais deduplicadas (arredondadas em 1e-4); índices a partir de 1. ``u``: coordenada
    de textura por canto (F, 3) de cada parte, gravada como ``vt u 0.5`` (deduplicada em 1e-4)."""
    lines = [f"# {line}" for line in header.splitlines()]
    v_off, all_n, all_t = 0, [], []
    face_lines = []
    n_index: dict[tuple, int] = {}
    t_index: dict[float, int] = {}
    v_lines = []
    for k, (verts, faces, normals) in enumerate(parts):
        v_lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
        tri_u = u[k] if u is not None else np.full(faces.shape, np.nan)
        for tri, tri_n, tu in zip(faces, normals, tri_u, strict=True):
            idx = []
            for vi, n, uc in zip(tri, tri_n, tu, strict=True):
                key = tuple(np.round(n, 4))
                if key not in n_index:
                    n_index[key] = len(all_n) + 1
                    all_n.append(key)
                vt = ""
                if u is not None:
                    tk = round(float(uc), 4)
                    if tk not in t_index:
                        t_index[tk] = len(all_t) + 1
                        all_t.append(tk)
                    vt = str(t_index[tk])
                idx.append(f"{vi + 1 + v_off}/{vt}/{n_index[key]}")
            face_lines.append("f " + " ".join(idx))
        v_off += len(verts)
    lines += v_lines + [f"vn {x:.4f} {y:.4f} {z:.4f}" for x, y, z in all_n]
    lines += [f"vt {t:.4f} 0.5" for t in all_t] + face_lines
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    from pxr import Usd

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workshop", type=Path, default=ROOT.parent / "Sim-to-Real-SO-101-Workshop")
    ap.add_argument("--out", type=Path, default=ROOT / "assets" / "vials_lite")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name, (usd, prims) in MESHES.items():
        src = args.workshop / USD_DIR / usd
        stage = Usd.Stage.Open(str(src))
        parts = [read_mesh(stage, p) for p in prims]
        u = None
        if name in SMOOTH_KNURL:
            parts = [(v2, f, n2) for v, f, n in parts for v2, n2 in [smooth_knurl(v, f, n)]]
            u = [knurl_u(*p) for p in parts]
        header = (
            f"Derivado de {USD_DIR.as_posix()}/{usd} ({', '.join(prims)}) do Sim-to-Real SO-101 Workshop,\n"
            "Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.\n"
            "Gerado por scripts/export_workshop_meshes.py; metros, frame do asset, só visual."
        )
        write_obj(args.out / name, parts, header, u)
        verts = np.concatenate([p[0] for p in parts])
        print(
            f"{name}: {len(verts)} vértices, {sum(len(p[1]) for p in parts)} triângulos,"
            f" caixa {verts.min(0).round(4).tolist()} .. {verts.max(0).round(4).tolist()}"
        )


if __name__ == "__main__":
    main()
