"""Extrai dos USDs do workshop o que a versão MuJoCo precisa: poses de câmeras, slots do rack,
extents e malhas. Roda só com ``usd-core`` (pip), sem Isaac Sim.

Pré-requisito: o repositório do workshop com os arquivos LFS baixados:
    git clone https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop.git
    cd Sim-to-Real-SO-101-Workshop
    git lfs pull --include="source/sim_to_real_so101/assets/usd/*"

Uso (uma linha; funciona em PowerShell e bash), com <usd> = <workshop>/source/sim_to_real_so101/assets/usd:
    python scripts/inspect_workshop_assets.py <usd> --out assets/workshop_extract.json

Saída: JSON com, para cada arquivo, os prims de interesse (câmeras, ``top_*`` do rack,
``camera_mount``) e sua transformação no frame do próprio arquivo (posição + quaternion wxyz).
Lembre que no Isaac esses arquivos são instanciados com offsets (spec.LIGHTSTUDIO_POS etc.).

Para malhas, além do ``extent`` autorado, o script calcula ``points_bbox`` a partir dos pontos: nos
assets reais o ``extent`` do frasco está desatualizado (dá ~0,7 mm) e não deve ser usado.
Payloads/referências que não abrem (ex.: a câmera D455, que vem de um USD remoto do Isaac) aparecem
em ``unresolved_payloads``.

Status: testado com um stage sintético (tests/test_inspect_usd.py) e com os assets reais do workshop
(commit ce807d9, no Windows e no Ubuntu, com os mesmos valores).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

INTERESTING = re.compile(r"(Camera|camera_mount|top_|rsd455|Mesh$|Body1)", re.IGNORECASE)


def inspect_stage(path: Path) -> dict:
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"não abriu: {path}")
    default_prim = stage.GetDefaultPrim()
    out: dict = {
        "file": path.name,
        "up_axis": UsdGeom.GetStageUpAxis(stage),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "default_prim": str(default_prim.GetPath()) if default_prim else None,
        "unresolved_payloads": [],
        "prims": [],
    }
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in stage.Traverse():
        if prim.HasAuthoredPayloads() and not prim.GetChildren():
            for spec in prim.GetPrimStack():
                for item in spec.payloadList.GetAddedOrExplicitItems():
                    out["unresolved_payloads"].append({"path": str(prim.GetPath()), "asset": item.assetPath})
        name = prim.GetName()
        is_root = prim.GetParent() == stage.GetPseudoRoot()
        if not (is_root or INTERESTING.search(name) or prim.IsA(UsdGeom.Camera)):
            continue
        entry: dict = {"path": str(prim.GetPath()), "type": prim.GetTypeName()}
        if prim.IsA(UsdGeom.Xformable):
            m = cache.GetLocalToWorldTransform(prim)
            xf = Gf.Transform(m)
            t = xf.GetTranslation()
            q = xf.GetRotation().GetQuat()
            entry["pos"] = [t[0], t[1], t[2]]
            entry["quat_wxyz"] = [q.GetReal(), *q.GetImaginary()]
            entry["scale"] = list(xf.GetScale())
        if prim.IsA(UsdGeom.Mesh):
            pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if pts:
                mat = np.array(m)  # convenção USD: vetor-linha, p_mundo = p · M
                world = np.asarray(pts, dtype=float) @ mat[:3, :3] + mat[3, :3]
                entry["points_bbox"] = [world.min(axis=0).tolist(), world.max(axis=0).tolist()]
                entry["n_points"] = len(pts)
        if prim.IsA(UsdGeom.Camera):
            cam = UsdGeom.Camera(prim)
            for attr in ("focalLength", "horizontalAperture", "verticalAperture", "clippingRange"):
                v = getattr(cam, f"Get{attr[0].upper()}{attr[1:]}Attr")().Get()
                entry[attr] = list(v) if hasattr(v, "__len__") else v
        if prim.IsA(UsdGeom.Boundable):
            ext = UsdGeom.Boundable(prim).GetExtentAttr().Get()
            if ext:
                entry["extent"] = [list(ext[0]), list(ext[1])]
        out["prims"].append(entry)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("usd_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("assets/workshop_extract.json"))
    args = ap.parse_args()
    files = sorted(p for p in args.usd_dir.glob("*.usd*") if p.stat().st_size > 1024)
    if not files:
        raise SystemExit("nenhum USD real encontrado — rodou `git lfs pull`? (ponteiros LFS têm ~130 bytes)")
    result = [inspect_stage(p) for p in files]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{len(result)} arquivos → {args.out}")


if __name__ == "__main__":
    main()
