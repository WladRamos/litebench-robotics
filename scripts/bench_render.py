"""Benchmark: quanto a sua máquina aguenta? Física + 2 câmeras 640×480 no MuJoCo.

Cena: SO-101 do MuJoCo Menagerie + um frasco (cilindro) e um rack (caixa) aproximados.
Mede steps de controle por segundo (1 step = 1/60 s simulado), com e sem render.

Uso (o clone do Menagerie é o mesmo nos dois sistemas):
    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git <menagerie>
    (sugestão: <menagerie> = assets/third_party/mujoco_menagerie, ignorado pelo git)

    Linux (bash):          MUJOCO_GL=egl uv run python scripts/bench_render.py --menagerie <menagerie>
                           (EGL não precisa de monitor; com monitor, sem MUJOCO_GL, usa GLFW)
    Windows (PowerShell):  uv run python scripts/bench_render.py --menagerie <menagerie>
                           (não defina MUJOCO_GL: o padrão no Windows é GLFW/WGL, sem janela visível)

O script imprime qual GPU está de fato renderizando (GL_RENDERER). Em notebooks com GPU híbrida
(NVIDIA Optimus), o Python costuma cair na GPU integrada sem avisar. Se aparecer Intel/AMD/llvmpipe
em vez de NVIDIA, force a RTX nas "Configurações de gráficos" do Windows (adicione o python.exe do
.venv como "Alto desempenho") ou no Painel de Controle da NVIDIA, e meça de novo.
Atenção com venvs do uv no Windows: o .venv\\Scripts\\python.exe é só um lançador; quem abre o contexto
OpenGL é o interpretador base (sys._base_executable, em %APPDATA%\\uv\\python\\...). Marque os dois
(e o destino da junção cpython-3.12-... → cpython-3.12.x-...) como "Alto desempenho".
No Linux com EGL, se aparecer Mesa/Intel: confira primeiro se o driver NVIDIA está carregado (nvidia-smi;
após atualizar o kernel é preciso reiniciar). No notebook Ubuntu, com o driver carregado, o EGL
escolheu a NVIDIA sozinho. Se não escolher na sua máquina, tente MUJOCO_EGL_DEVICE_ID=<n> ou
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json (não testados aqui).

Referência medida num contêiner Linux SEM GPU (2 núcleos, MuJoCo 3.14, OSMesa/llvmpipe):
    física pura:            ~20–28 mil steps/s (1 thread)
    física + 2 câmeras:     1,8–1,9 steps/s   ← render por software; é o limite inferior
Medido no notebook Windows 11 (RTX 4050 Laptop, driver 596.08, MuJoCo 3.14, GLFW/WGL):
    física pura:            ~37–41 mil steps/s
    física + 2 câmeras:     395–434 steps/s na RTX; 7,7 steps/s se cair na Intel integrada
Medido no notebook Ubuntu 24.04 (RTX 5050 Laptop 8 GB, driver 595.91.07, MuJoCo 3.14, EGL):
    física pura:            ~48–49 mil steps/s
    física + 2 câmeras:     685–706 steps/s na RTX; 44 steps/s na Intel integrada (Mesa)
Para comparar, anote o número da sua máquina com o SO e o GL_RENDERER.
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import mujoco
import numpy as np

EXTRA = """
  <worldbody>
    <camera name="external" pos="0.55 0.0 0.45" xyaxes="0 1 0 -0.6 0 0.8"/>
    <body name="vial" pos="0.23 0 0.05"><freejoint/>
      <geom type="cylinder" size="0.008 0.04" mass="0.02" rgba="0.8 0.8 1 1"/></body>
    <body name="rack" pos="0.18 0.12 0.03">
      <geom type="box" size="0.06 0.06 0.03" rgba="1 0.9 0.1 1"/></body>
  </worldbody>
"""


def gl_renderer_name(renderer: mujoco.Renderer) -> str:
    """Nome da GPU/driver que o contexto OpenGL do MuJoCo está usando (melhor esforço)."""
    try:
        from OpenGL import GL  # PyOpenGL é dependência do pacote mujoco

        renderer._gl_context.make_current()  # atributo interno; pode mudar entre versões
        name = GL.glGetString(GL.GL_RENDERER)
        vendor = GL.glGetString(GL.GL_VENDOR)
        return f"{name.decode()} ({vendor.decode()})"
    except Exception as e:  # noqa: BLE001 — diagnóstico, não pode derrubar o benchmark
        return f"desconhecido ({type(e).__name__})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--menagerie", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    model_dir = args.menagerie / "robotstudio_so101"
    xml = (model_dir / "scene.xml").read_text(encoding="utf-8").replace("</mujoco>", EXTRA + "</mujoco>")
    tmp = model_dir / "_bench_scene.xml"  # precisa ficar ao lado dos meshes
    tmp.write_text(xml, encoding="utf-8")
    try:
        m = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    d = mujoco.MjData(m)
    n_sub = max(1, round((1 / 60) / m.opt.timestep))
    print(f"{platform.system()} {platform.release()} | Python {platform.python_version()}")
    print(f"MuJoCo {mujoco.__version__} | timestep {m.opt.timestep} | {n_sub} substeps/step de controle")

    t = time.perf_counter()
    for i in range(args.steps * 10):
        d.ctrl[:] = 0.3 * np.sin(i / 30.0)
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
    dt = time.perf_counter() - t
    print(f"física pura: {args.steps * 10 / dt:,.0f} steps/s")

    r = mujoco.Renderer(m, args.height, args.width)
    gpu = gl_renderer_name(r)
    print(f"GL_RENDERER: {gpu}")
    if "nvidia" not in gpu.lower():
        print("AVISO: o render não está na GPU NVIDIA; veja a docstring antes de anotar o número.")
    t = time.perf_counter()
    for i in range(args.steps):
        d.ctrl[:] = 0.3 * np.sin(i / 30.0)
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
        for cam in ("wrist_cam", "external"):
            r.update_scene(d, camera=cam)
            r.render()
    dt = time.perf_counter() - t
    rate = args.steps / dt
    backend = os.environ.get("MUJOCO_GL", "padrão do sistema")
    print(f"física + 2 câmeras {args.width}x{args.height} ({backend}): {rate:,.1f} steps/s")
    ep_s = 450 / rate
    print(f"→ episódio de 450 steps ≈ {ep_s:.1f} s (sem política); 20 episódios ≈ {20 * ep_s / 60:.1f} min")


if __name__ == "__main__":
    main()
