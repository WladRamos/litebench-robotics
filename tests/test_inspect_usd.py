import importlib.util
import sys
from pathlib import Path

import pytest

pxr = pytest.importorskip("pxr")


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "inspect_workshop_assets.py"
    spec = importlib.util.spec_from_file_location("inspect_workshop_assets", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_extracts_camera_and_slot_world_poses(tmp_path):
    from pxr import Gf, Usd, UsdGeom

    f = tmp_path / "synthetic.usda"
    stage = Usd.Stage.CreateNew(str(f))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/LightBox")
    root.AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.0))
    mount = UsdGeom.Xform.Define(stage, "/LightBox/camera_mount")
    mount.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.5, 0.25))
    cam = UsdGeom.Camera.Define(stage, "/LightBox/camera_mount/Camera_OV9782_Right")
    cam.GetFocalLengthAttr().Set(13.5)
    slot = UsdGeom.Xform.Define(stage, "/Rack/Body1/Mesh/top_0")
    slot.AddTranslateOp().Set(Gf.Vec3d(0.03, 0.03, 0.1))
    vial = UsdGeom.Xform.Define(stage, "/Vial")
    vial.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.0))
    mesh = UsdGeom.Mesh.Define(stage, "/Vial/Mesh")
    mesh.GetPointsAttr().Set([(-0.01, -0.01, 0.0), (0.01, 0.01, 0.08)])
    mesh.GetExtentAttr().Set([(-0.001, -0.001, 0.0), (0.001, 0.001, 0.001)])  # extent desatualizado
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()

    out = _load_script().inspect_stage(f)
    by_path = {p["path"]: p for p in out["prims"]}
    c = by_path["/LightBox/camera_mount/Camera_OV9782_Right"]
    assert c["pos"] == pytest.approx([1.0, 0.5, 0.25])
    assert c["focalLength"] == pytest.approx(13.5)
    assert by_path["/Rack/Body1/Mesh/top_0"]["pos"] == pytest.approx([0.03, 0.03, 0.1])
    assert out["default_prim"] == "/LightBox"
    lo, hi = by_path["/Vial/Mesh"]["points_bbox"]  # vem dos pontos (com a translação), não do extent
    assert lo == pytest.approx([-0.01, -0.01, 1.0])
    assert hi == pytest.approx([0.01, 0.01, 1.08])
