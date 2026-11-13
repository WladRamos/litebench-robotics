"""Gravador do Isaac: roda o ``lerobot_eval.py`` do workshop, sem mudanças, registrando cada episódio.

Objetivo: fases e falhas no Isaac pelo evalkit, a cena inicial de cada
episódio para avaliar as mesmas cenas no Vials-Lite (comparação pareada) e as ações para repeti-las aqui
em malha aberta. Roda no Python do ambiente do Isaac e só usa NumPy e o próprio Isaac (não importa o
``vials_lite``); a conversão para o evalkit e para ``scenes.jsonl`` é feita depois, nesta máquina.

Como funciona: troca ``gymnasium.make`` por uma versão que envolve o ambiente criado num ``Recorder``
(``gym.Wrapper``) e executa o ``lerobot_eval.py`` com ``runpy``, com os argumentos dados depois de ``--``.
Só lê o estado; não consome números sorteados, então a mesma semente dá os mesmos episódios.

* Cada ``reset()`` do cliente abre um episódio e grava a cena inicial: poses do rack e dos frascos, juntas,
  e os atributos do USD que os eventos de reset do workshop sorteiam (``mdp/resets.py``): cor do robô,
  giro do tapete, focal da câmera de punho, pose do ``camera_mount`` e das câmeras, exposição da lightbox
  e, na DR, o domo (exposição, temperatura, HDRI e giro).
* Cada ``step()`` grava a ação (radianos, como o cliente manda) e o estado depois do step.
* O Isaac Lab reinicia sozinho, dentro do ``step``, o ambiente que terminou: quando o ``step`` retorna, o
  estado final já é o da cena seguinte (que o cliente descarta com outro ``reset()``). Por isso o gravador
  também envolve ``_reset_idx`` e guarda o estado logo antes do reinício; no step que termina, é esse
  estado que vai para o registro.
* Os termos ``subtask_terms`` da observação (``vial_grasped``, ``vial_placed``) também; no step que
  termina eles já são da cena seguinte e ficam NaN.
* Sucesso, como no cliente: ``terminated`` e não ``truncated``.

Saída em ``--out``: ``meta.json`` (nomes das juntas e dos corpos, argumentos), ``episodes.jsonl`` (uma
linha por episódio: sucesso, steps, cena inicial) e ``ep_NNNN.npz`` (séries por step). Cada episódio é
gravado assim que termina: o cliente encerra o processo sem voltar daqui.

Com ``--frames N``, grava também as imagens RGB que a política recebe (grupo ``visual`` da observação:
câmera de punho e externa) nos N primeiros episódios, em ``videos/ep_NNNN_<câmera>.mp4``, para ilustrar
os resultados. Usa o ``av`` do ambiente do Isaac; sem ele, avisa e segue sem vídeo. O quadro do
step que termina já é da cena seguinte e fica de fora.

Com ``--vial-sdf-resolution N``, os frascos usam uma cópia do ``Vial_opaque.usda`` com
``physxSDFMeshCollision:sdfResolution = N`` (o workshop usa 20), gravada ao lado do original como
``Vial_opaque.sdfN.usda``; o ``meta.json`` guarda a resolução lida no stage depois de criar o ambiente.
Sem a opção, nada muda.

Uso (na máquina do Isaac; ``scripts/isaac_eval_cells.sh`` já chama assim):
    PYTHONUNBUFFERED=1 $ISAAC_PY scripts/isaac_recorder.py --out outputs/isaac/<nome>/Eval.rec \\
        --script $WORKSHOP/source/sim_to_real_so101/scripts/lerobot_eval.py -- \\
        --task Lerobot-So101-Teleop-Vials-To-Rack-Eval --num_envs 1 --headless --seed 1984 ...
    (com ``FRAMES=5`` no ``isaac_eval_cells.sh``, acrescenta ``--frames 5`` antes do ``--script``)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ENV0 = "/World/envs/env_0"
VIALS = ("vial_1", "vial_2", "vial_3")
RACK = "rack_left"
CAMERAS = ("camera_ego", "camera_external_D455")
# atributos do USD sorteados pelos eventos de reset (mdp/resets.py e tasks/*_env_cfg.py do workshop)
USD_ATTRS: dict[str, tuple[str, str]] = {
    "robot_rgb": (f"{ENV0}/Robot/Looks/material_a_3d_printed/Shader", "inputs:diffuse_color_constant"),
    "mat_orient": (f"{ENV0}/Mat", "xformOp:orient"),
    "ego_focal_mm": (f"{ENV0}/Robot/gripper/gripper_cam", "focalLength"),
    "ego_vertical_aperture_mm": (f"{ENV0}/Robot/gripper/gripper_cam", "verticalAperture"),
    "ext_mount_translate": (f"{ENV0}/LightStudio/LightBox/camera_mount", "xformOp:translate"),
    "ext_mount_orient": (f"{ENV0}/LightStudio/LightBox/camera_mount", "xformOp:orient"),
    "rect_exposure": (f"{ENV0}/LightStudio/LightBox/RectLight", "inputs:exposure"),
    "dome_exposure": ("/World/sky_light", "inputs:exposure"),
    "dome_temperature_k": ("/World/sky_light", "inputs:colorTemperature"),
    "dome_texture": ("/World/sky_light", "inputs:texture:file"),
    "dome_orient": ("/World/sky_light", "xformOp:orient"),
}


def to_np(x: Any) -> np.ndarray:
    """Tensor do PyTorch (ou array) → ``np.ndarray`` float64."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def rgb_frames(obs: Any) -> dict[str, np.ndarray]:
    """Imagens RGB do 1º ambiente no grupo ``visual`` da observação, em ``uint8`` (H, W, 3)."""
    vis = obs.get("visual") if isinstance(obs, dict) else None
    out = {}
    for k, v in (vis or {}).items():
        if not k.startswith("rgb"):
            continue
        a = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        a = a[0] if a.ndim == 4 else a
        if a.dtype != np.uint8:
            a = np.clip(a * 255 if a.max() <= 1.0 else a, 0, 255).astype(np.uint8)
        out[k.removeprefix("rgb_")] = a[..., :3]
    return out


class VideoWriter:
    """Um MP4 por câmera, escrito quadro a quadro (H.264 pelo ``av``)."""

    def __init__(self, prefix: Path, fps: int) -> None:
        import av

        self._av, self.prefix, self.fps, self._out = av, prefix, fps, {}

    def add(self, frames: dict[str, np.ndarray]) -> None:
        for cam, f in frames.items():
            if cam not in self._out:
                c = self._av.open(str(self.prefix) + f"_{cam}.mp4", mode="w")
                st = c.add_stream("libx264", rate=self.fps)
                st.height, st.width = f.shape[0] - f.shape[0] % 2, f.shape[1] - f.shape[1] % 2
                st.pix_fmt, st.options = "yuv420p", {"crf": "20"}
                self._out[cam] = (c, st)
            c, st = self._out[cam]
            f = np.ascontiguousarray(f[: st.height, : st.width])
            vf = self._av.VideoFrame.from_ndarray(f, format="rgb24")
            for pkt in st.encode(vf):
                c.mux(pkt)

    def close(self) -> None:
        for c, st in self._out.values():
            for pkt in st.encode():
                c.mux(pkt)
            c.close()
        self._out = {}


def usd_value(v: Any) -> Any:
    """Valor de um atributo do USD → tipo do JSON (quaternions do Gf em wxyz)."""
    if v is None:
        return None
    if hasattr(v, "GetReal") and hasattr(v, "GetImaginary"):
        im = v.GetImaginary()
        return [float(v.GetReal()), float(im[0]), float(im[1]), float(im[2])]
    if hasattr(v, "path"):  # Sdf.AssetPath
        return str(v.path)
    if isinstance(v, (bool, int, float, str)):
        return v
    try:
        return [float(c) for c in v]
    except TypeError:
        return str(v)


def read_usd(stage: Any) -> dict[str, Any]:
    """Atributos de ``USD_ATTRS``; o que não existir (o domo na ``Eval``) fica ``None``."""
    out: dict[str, Any] = {}
    for key, (path, attr) in USD_ATTRS.items():
        try:
            prim = stage.GetPrimAtPath(path)
            a = prim.GetAttribute(attr) if prim and prim.IsValid() else None
            out[key] = usd_value(a.Get()) if a and a.IsValid() else None
        except Exception as e:  # noqa: BLE001 - o gravador não pode derrubar a avaliação
            out[key] = None
            out.setdefault("_errors", []).append(f"{key}: {e!r}")
    return out


def yaw_from_quat(q: list[float] | None) -> float | None:
    """Giro em z de um quaternion wxyz."""
    if not isinstance(q, list) or len(q) != 4:
        return None
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def read_state(env: Any) -> dict[str, np.ndarray]:
    """Estado privilegiado do ambiente 0, com as posições no frame do ambiente (sem ``env_origins``)."""
    scene = env.scene
    origin = to_np(scene.env_origins)[0]
    robot = scene["robot"]
    s = {
        "joint_pos": to_np(robot.data.joint_pos)[0],
        "body_pos": to_np(robot.data.body_pos_w)[0] - origin,
        "body_quat": to_np(robot.data.body_quat_w)[0],
        "vial_pos": np.stack([to_np(scene[v].data.root_pos_w)[0] - origin for v in VIALS]),
        "vial_quat": np.stack([to_np(scene[v].data.root_quat_w)[0] for v in VIALS]),
        "rack_pos": to_np(scene[RACK].data.root_pos_w)[0] - origin,
        "rack_quat": to_np(scene[RACK].data.root_quat_w)[0],
    }
    # força mandíbula ↔ cada frasco (N), a que os termos de sucesso do workshop usam
    fm = to_np(scene["contact_grasp"].data.force_matrix_w)[0]  # (corpos, filtros, 3)
    s["jaw_force"] = np.linalg.norm(fm, axis=-1).sum(axis=0)
    return s


def read_cameras(env: Any) -> dict[str, Any]:
    """Pose no mundo (sem ``env_origins``) e intrínsecos das câmeras, melhor esforço."""
    out: dict[str, Any] = {}
    origin = to_np(env.scene.env_origins)[0]
    for name in CAMERAS:
        try:
            data = env.scene[name].data
            cam = {"pos": (to_np(data.pos_w)[0] - origin).tolist()}
            for q in ("quat_w_world", "quat_w_ros", "quat_w_opengl"):
                if getattr(data, q, None) is not None:
                    cam[q] = to_np(getattr(data, q))[0].tolist()
            if getattr(data, "intrinsic_matrices", None) is not None:
                cam["K"] = to_np(data.intrinsic_matrices)[0].tolist()
            out[name] = cam
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": repr(e)}
    return out


def default_stage() -> Any:
    import omni.usd

    return omni.usd.get_context().get_stage()


def camera_mount_default() -> Any:
    """Pose padrão do ``camera_mount`` lida pelo workshop (``_default_poses_cache``), se já existir."""
    try:
        from sim_to_real_so101.mdp import resets

        return {k: {kk: list(vv) for kk, vv in v.items()} for k, v in resets._default_poses_cache.items()}
    except Exception:  # noqa: BLE001
        return None


SDF_RESOLUTION_ATTR = "physxSDFMeshCollision:sdfResolution"


def usd_with_sdf_resolution(usd_path: Path, resolution: int) -> Path:
    """Cópia de um ``.usda`` com todo ``sdfResolution`` trocado por ``resolution``, na mesma pasta (as
    texturas são caminhos relativos). Erro se o arquivo não tiver o atributo."""
    import re

    usd_path = Path(usd_path)
    text, n = re.subn(rf"({re.escape(SDF_RESOLUTION_ATTR)} = )\d+", rf"\g<1>{resolution}",
                      usd_path.read_text(encoding="utf-8"))
    if n == 0:
        raise ValueError(f"{usd_path} não tem {SDF_RESOLUTION_ATTR}")
    out = usd_path.with_name(f"{usd_path.stem}.sdf{resolution}{usd_path.suffix}")
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")  # clientes em paralelo escrevem o mesmo arquivo
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out)
    return out


def set_vial_sdf_resolution(cfg: Any, resolution: int) -> list[str]:
    """Aponta o ``usd_path`` dos frascos da configuração da cena para a cópia com outra resolução do SDF."""
    paths = []
    for name in VIALS:
        spawn = getattr(cfg.scene, name).spawn
        spawn.usd_path = str(usd_with_sdf_resolution(Path(spawn.usd_path), resolution))
        paths.append(spawn.usd_path)
    return paths


def read_sdf_resolutions(stage: Any) -> dict[str, list[int]]:
    """``sdfResolution`` de cada prim debaixo de cada frasco do ambiente 0, lido no stage."""
    from pxr import Usd

    out = {}
    for name in VIALS:
        root = stage.GetPrimAtPath(f"{ENV0}/{name.capitalize()}")
        vals = []
        for prim in Usd.PrimRange(root) if root and root.IsValid() else []:
            a = prim.GetAttribute(SDF_RESOLUTION_ATTR)
            if a and a.IsValid() and a.Get() is not None:
                vals.append(int(a.Get()))
        out[name] = vals
    return out


class Recorder:
    """Envolve o ambiente do cliente (``gym.Wrapper`` sem herdar, para não exigir a mesma versão)."""

    def __init__(self, env: Any, out: Path, stage_fn: Callable[[], Any] = default_stage,
                 state_fn: Callable[[Any], dict] = read_state,
                 cameras_fn: Callable[[Any], dict] = read_cameras, frames: int = 0,
                 meta_extra: dict | None = None) -> None:
        self.env = env
        self.frames = frames
        self._video: VideoWriter | None = None
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self._stage_fn, self._state_fn, self._cameras_fn = stage_fn, state_fn, cameras_fn
        self.episode = -1
        self._steps: list[dict] | None = None
        self._scene: dict | None = None
        self._in_step = False
        self._pre_reset: dict | None = None
        self._t0 = time.perf_counter()
        base = env.unwrapped
        orig = base._reset_idx

        def _reset_idx(*a, **kw):  # estado final do episódio, antes do reinício automático
            if self._in_step:
                self._pre_reset = self._safe_state()
            return orig(*a, **kw)

        base._reset_idx = _reset_idx
        meta = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": sys.argv, **(meta_extra or {})}
        try:
            robot = base.scene["robot"]
            meta["joint_names"] = list(robot.data.joint_names)
            meta["body_names"] = list(robot.data.body_names)
        except Exception as e:  # noqa: BLE001
            meta["error"] = repr(e)
        self.fps = round(1 / base.step_dt) if getattr(base, "step_dt", None) else 30
        meta["frames"], meta["video_fps"] = frames, self.fps
        (self.out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    def __getattr__(self, name: str) -> Any:  # unwrapped, action_space, close, ...
        return getattr(self.env, name)

    def _safe_state(self) -> dict | None:
        try:
            return self._state_fn(self.env.unwrapped)
        except Exception as e:  # noqa: BLE001
            print(f"[isaac_recorder] estado não lido: {e!r}", flush=True)
            return None

    def reset(self, *args, **kwargs):
        if self._steps:  # episódio interrompido sem terminar (reset manual)
            self._flush(success=False, truncated=False, interrupted=True)
        result = self.env.reset(*args, **kwargs)
        self.episode += 1
        self._steps = []
        if self.episode < self.frames:
            try:
                (self.out / "videos").mkdir(exist_ok=True)
                self._video = VideoWriter(self.out / "videos" / f"ep_{self.episode:04d}", self.fps)
                obs = result[0] if isinstance(result, tuple) else result
                self._video.add(rgb_frames(obs))
            except Exception as e:  # noqa: BLE001
                print(f"[isaac_recorder] sem vídeo: {e!r}", flush=True)
                self._video, self.frames = None, 0
        base = self.env.unwrapped
        try:
            usd = read_usd(self._stage_fn())
        except Exception as e:  # noqa: BLE001
            usd = {"_errors": [repr(e)]}
        usd["mat_yaw_raw"] = yaw_from_quat(usd.get("mat_orient"))
        state0 = self._safe_state()
        self._scene = {
            "usd": usd,
            "cameras": self._cameras_fn(base),
            "camera_mount_default": camera_mount_default(),
            "state0": {k: v.tolist() for k, v in state0.items()} if state0 else None,
        }
        return result

    def step(self, action):
        self._in_step, self._pre_reset = True, None
        try:
            obs, rew, terminated, truncated, info = self.env.step(action)
        finally:
            self._in_step = False
        term, trunc = bool(to_np(terminated).any()), bool(to_np(truncated).any())
        state = self._pre_reset if (term or trunc) and self._pre_reset is not None else self._safe_state()
        rec = {"action": to_np(action).reshape(-1, to_np(action).shape[-1])[0]}
        if state:
            rec.update(state)
        try:
            sub = obs.get("subtask_terms") if isinstance(obs, dict) else None
            if sub:
                for k, v in sub.items():  # no step que termina, a observação já é da cena seguinte
                    rec[k] = np.nan if term or trunc else to_np(v).reshape(-1)[0]
        except Exception:  # noqa: BLE001
            pass
        if self._video is not None and not (term or trunc):
            try:
                self._video.add(rgb_frames(obs))
            except Exception as e:  # noqa: BLE001
                print(f"[isaac_recorder] vídeo interrompido: {e!r}", flush=True)
                self._video, self.frames = None, 0
        if self._steps is not None:
            self._steps.append(rec)
            if term or trunc:
                self._flush(success=term and not trunc, truncated=trunc)
        return obs, rew, terminated, truncated, info

    def _flush(self, success: bool, truncated: bool, interrupted: bool = False) -> None:
        steps, self._steps = self._steps or [], None
        if self._video is not None:
            self._video.close()
            self._video = None
        keys = sorted({k for s in steps for k in s})
        arrays = {k: np.stack([s[k] for s in steps if k in s]) for k in keys}
        np.savez_compressed(self.out / f"ep_{self.episode:04d}.npz", **arrays)
        row = {"episode": self.episode, "success": bool(success), "truncated": bool(truncated),
               "interrupted": interrupted, "steps": len(steps),
               "wall_s": round(time.perf_counter() - self._t0, 1), "scene": self._scene}
        with (self.out / "episodes.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        self._t0 = time.perf_counter()


def main() -> None:
    if "--" not in sys.argv:
        sys.exit("uso: isaac_recorder.py --out PASTA --script lerobot_eval.py -- <argumentos do cliente>")
    i = sys.argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--script", type=Path, required=True, help="o lerobot_eval.py do workshop")
    ap.add_argument("--frames", type=int, default=0, help="grava as câmeras dos N primeiros episódios")
    ap.add_argument("--vial-sdf-resolution", type=int, default=None,
                    help="resolução do SDF de colisão dos frascos (o workshop usa 20)")
    args = ap.parse_args(sys.argv[1:i])
    client_argv = sys.argv[i + 1 :]

    import runpy

    import gymnasium

    make = gymnasium.make

    def recording_make(*a, **kw):
        extra = {}
        if args.vial_sdf_resolution is not None:
            extra["vial_usd"] = set_vial_sdf_resolution(kw["cfg"], args.vial_sdf_resolution)
        env = make(*a, **kw)
        if extra:
            try:
                extra["vial_sdf_resolution_stage"] = read_sdf_resolutions(default_stage())
            except Exception as e:  # noqa: BLE001
                extra["vial_sdf_resolution_stage"] = repr(e)
            print(f"[isaac_recorder] frascos: {extra}", flush=True)
        print(f"[isaac_recorder] gravando em {args.out}", flush=True)
        return Recorder(env, args.out, frames=args.frames, meta_extra=extra)

    gymnasium.make = recording_make
    sys.argv = [str(args.script), *client_argv]
    runpy.run_path(str(args.script), run_name="__main__")


if __name__ == "__main__":
    main()
