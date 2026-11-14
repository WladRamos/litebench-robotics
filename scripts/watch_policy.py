"""Assiste a uma política no Vials-Lite: episódios consecutivos, até encerrar.

Cada episódio segue o protocolo do ``evalkit.runner`` (aquecimento, 450 steps, término no sucesso), com
as sementes ``seed``, ``seed + 1``, ... A cada episódio o terminal mostra a semente, o resultado, a
fase máxima e a categoria de falha. Para repetir um episódio visto aqui com o relatório completo:
``python -m evalkit.runner --policy <política> --seed <semente> --episodes 1 --video 1``.

Não grava nada: é para ver. Para medir, use o runner.

``--policy scripted`` usa a demo roteirizada (``vials_lite.scripted``), sem LeRobot nem política
treinada. Sem ``--seed``, a primeira semente é sorteada.

Dois modos:

* **Janela (padrão):** o viewer do MuJoCo, na máquina que roda o script (precisa de monitor). Abre na
  câmera externa (a que a política vê; ``--camera ego`` ou ``--camera free`` abrem em outra). Todas as
  câmeras do ambiente estão na lista Camera do painel esquerdo (seção Rendering), e ``[`` e ``]``
  percorrem a mesma lista; Esc passa para a câmera livre (arrastar gira, botão direito desloca, roda
  aproxima) e a barra de espaço pausa. Fechar a janela ou Ctrl+C encerra.
* **Navegador (``--stream PORTA``):** para uma máquina remota. O render é feito sem janela (EGL) e as
  imagens são servidas em ``http://localhost:PORTA``, só para a própria máquina; de outra, abra um
  túnel SSH (``ssh -L PORTA:localhost:PORTA <máquina>``) e acesse o mesmo endereço. A página escolhe a
  câmera (as do ambiente ou uma livre, que gira arrastando e aproxima com a roda), pausa e mostra o
  resultado de cada episódio. Ctrl+C no terminal encerra. Codifica JPEG com o Pillow, que vem com o
  extra ``lerobot`` (com ``--policy scripted`` sem esse extra: ``uv run --with pillow ...``).

Uso::

    uv run python scripts/watch_policy.py --policy scripted
    uv run python scripts/watch_policy.py --policy arjunjuneja/act_vials_50k --variant dr-eval
    uv run python scripts/watch_policy.py --policy arjunjuneja/act_vials_50k --stream 8000
    uv run python scripts/watch_policy.py --policy groot://localhost:5555 --action-horizon 16 \\
        --rename-map '{"external_D455": "front", "ego": "wrist"}'
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from evalkit.runner import run_episode
from vials_lite import spec
from vials_lite.env import VARIANTS, VialsLiteEnv
from vials_lite.scripted import ScriptedPickPlace

FREE_CAMERA = "livre"
# começo da câmera livre: de cima, pelo lado do rack, dentro da caixa de luz (de fora, as paredes tapam)
FREE_LOOKAT, FREE_AZIMUTH, FREE_ELEVATION, FREE_DISTANCE = (0.15, 0.0, 0.03), -90.0, -55.0, 0.58


class ScriptedPolicy:
    """A demo roteirizada na interface de política do runner (``reset`` + ``act``)."""

    name = "scripted"

    def __init__(self, env: VialsLiteEnv) -> None:
        self.env = env
        self._actions = None

    def reset(self) -> None:
        # chamado pelo runner depois do reset do ambiente: a demo planeja a partir da cena sorteada
        self._actions = ScriptedPickPlace(self.env).actions()

    def act(self, observation, instruction) -> np.ndarray:
        return next(self._actions)


class ViewerClosed(Exception):
    pass


class Pacer:
    """Segura o laço em ``speed`` × tempo real e espera enquanto ``paused``. Os modos recebem um frame
    por step do ``run_episode`` (o parâmetro ``frames``) em ``append``."""

    def __init__(self, speed: float) -> None:
        self.paused = False
        self.dt = 1.0 / (spec.TIMING.control_hz * speed)
        self._next = time.perf_counter()

    def wait(self) -> None:
        self._next += self.dt
        delay = self._next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:  # política mais lenta que o tempo real: não acumula atraso
            self._next = time.perf_counter()

    def hold_while_paused(self, refresh=None) -> None:
        while self.paused:
            time.sleep(0.05)
            if refresh is not None and not refresh():
                return
        self._next = time.perf_counter()


class ViewerSync(Pacer):
    """Modo janela: atualiza o viewer do MuJoCo a cada step; a barra de espaço pausa."""

    def __init__(self, speed: float) -> None:
        super().__init__(speed)
        self.viewer = None

    def key(self, keycode: int) -> None:
        if keycode == ord(" "):
            self.paused = not self.paused

    def _refresh(self) -> bool:
        self.viewer.sync()
        return self.viewer.is_running()

    def append(self, _frame) -> None:
        if not self.viewer.is_running():
            raise ViewerClosed
        self.viewer.sync()
        self.hold_while_paused(self._refresh)
        self.wait()

    def episode_done(self, line: str) -> None:
        pass


class StreamSink(Pacer):
    """Modo navegador: renderiza a câmera escolhida na página e guarda o último frame para o servidor.

    O render fica todo nesta thread (a do laço), porque o contexto OpenGL não troca de thread; o servidor
    só lê o frame pronto. Renderiza um step sim, um não (30 quadros por segundo)."""

    def __init__(self, env: VialsLiteEnv, speed: float, camera: str) -> None:
        import mujoco

        super().__init__(speed)
        self._mj, self.env = mujoco, env
        self.cameras = [env.model.camera(i).name for i in range(env.model.ncam)] + [FREE_CAMERA]
        if camera == "free":
            camera = FREE_CAMERA
        if camera not in self.cameras:
            raise ValueError(f"câmera {camera!r} não existe (use {', '.join(self.cameras)})")
        self.camera = camera
        self.free = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(env.model, self.free)
        self.free.lookat[:] = FREE_LOOKAT
        self.free.azimuth, self.free.elevation = FREE_AZIMUTH, FREE_ELEVATION
        self.free.distance = FREE_DISTANCE
        self.renderer = mujoco.Renderer(env.model, 480, 640)
        self.lock = threading.Lock()
        self.new_frame = threading.Condition(self.lock)
        self.frame: np.ndarray | None = None
        self.frame_id = 0
        self.step = 0
        self.history: list[str] = []

    def control(self, query: dict[str, list[str]]) -> None:
        """Comandos da página: ``pause``, ``camera=<nome>``, ``orbit=dx,dy`` e ``zoom=<passos>``."""
        with self.lock:
            if "pause" in query:
                self.paused = not self.paused
            if query.get("camera", [""])[0] in self.cameras:
                self.camera = query["camera"][0]
            if "orbit" in query:
                dx, dy = (float(v) for v in query["orbit"][0].split(","))
                self.free.azimuth -= 0.3 * dx
                self.free.elevation = float(np.clip(self.free.elevation - 0.3 * dy, -89, 89))
            if "zoom" in query:
                self.free.distance *= 1.1 ** float(query["zoom"][0])

    def status(self) -> dict:
        with self.lock:
            return {"camera": self.camera, "cameras": self.cameras, "paused": self.paused,
                    "step": self.step, "history": self.history[-12:]}

    def _render(self) -> None:
        with self.lock:
            cam = self.free if self.camera == FREE_CAMERA else self.camera
            self.renderer.update_scene(self.env.data, camera=cam)
            self.frame = self.renderer.render().copy()
            self.frame_id += 1
            self.new_frame.notify_all()

    def append(self, _frame) -> None:
        self.step += 1
        if self.step % 2 == 0:
            self._render()
        if self.paused:
            # parado, a imagem continua respondendo à troca de câmera e à câmera livre
            def refresh() -> bool:
                self._render()
                return True

            self.hold_while_paused(refresh)
        self.wait()

    def episode_done(self, line: str) -> None:
        with self.lock:
            self.history.append(line)
            self.step = 0

    def close(self) -> None:
        self.renderer.close()


PAGE = """<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Vials-Lite</title>
<style>
  body { margin: 0; padding: 16px; background: #111; color: #ddd; font: 14px system-ui, sans-serif; }
  .bar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }
  select, button { font: inherit; padding: 4px 10px; }
  img { width: 100%; max-width: 960px; display: block; background: #000; cursor: grab;
        user-select: none; -webkit-user-drag: none; }
  pre { margin: 8px 0 0; font-size: 13px; color: #aaa; }
  .hint { color: #888; }
</style></head><body>
<div class="bar">
  <label>Câmera <select id="cam"></select></label>
  <button id="pause">Pausar</button>
  <span id="step"></span>
  <span class="hint" id="hint">na câmera livre: arraste para girar, roda para aproximar</span>
</div>
<img id="view" src="/stream" alt="render do Vials-Lite">
<pre id="hist"></pre>
<script>
const cam = document.getElementById("cam"), pause = document.getElementById("pause");
const view = document.getElementById("view");
const send = (q) => fetch("/control?" + q);
cam.onchange = () => send("camera=" + encodeURIComponent(cam.value));
pause.onclick = () => send("pause=1");
let drag = null;
view.onmousedown = (e) => { drag = [e.clientX, e.clientY]; e.preventDefault(); };
window.onmouseup = () => { drag = null; };
window.onmousemove = (e) => {
  if (!drag || cam.value !== "livre") return;
  send(`orbit=${e.clientX - drag[0]},${e.clientY - drag[1]}`); drag = [e.clientX, e.clientY];
};
view.onwheel = (e) => {
  if (cam.value === "livre") { send("zoom=" + Math.sign(e.deltaY)); e.preventDefault(); }
};
async function poll() {
  try {
    const s = await (await fetch("/status")).json();
    if (cam.options.length !== s.cameras.length)
      cam.innerHTML = s.cameras.map(c => `<option>${c}</option>`).join("");
    if (document.activeElement !== cam) cam.value = s.camera;
    pause.textContent = s.paused ? "Continuar" : "Pausar";
    document.getElementById("step").textContent = `step ${s.step}/450`;
    document.getElementById("hist").textContent = s.history.slice().reverse().join("\\n");
  } catch (e) {}
  setTimeout(poll, 500);
}
poll();
</script></body></html>
"""


def serve(sink: StreamSink, port: int) -> ThreadingHTTPServer:
    """Servidor HTTP só em 127.0.0.1 (de fora, por túnel SSH): a página, o MJPEG e os comandos."""
    from PIL import Image

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # sem uma linha por requisição no terminal
            pass

        def _send(self, body: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif url.path == "/status":
                self._send(json.dumps(sink.status()).encode(), "application/json")
            elif url.path == "/control":
                sink.control(parse_qs(url.query))
                self._send(b"ok", "text/plain")
            elif url.path == "/stream":
                self._stream()
            else:
                self.send_error(404)

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = -1
            try:
                while True:
                    with sink.new_frame:
                        sink.new_frame.wait_for(lambda last=last: sink.frame_id != last, timeout=1.0)
                        if sink.frame is None or sink.frame_id == last:
                            continue
                        frame, last = sink.frame, sink.frame_id
                    buf = io.BytesIO()
                    Image.fromarray(frame).save(buf, format="JPEG", quality=85)
                    jpg = buf.getvalue()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # a página foi fechada

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def load_policy(args, ap, env):
    if args.policy == "scripted":
        return ScriptedPolicy(env), 0
    if args.policy.startswith("groot://"):
        from evalkit.groot_bridge import GrootRemotePolicy, parse_groot_url

        if args.action_horizon is None:
            ap.error("--policy groot://... exige --action-horizon")
        rename_map = json.loads(args.rename_map) if args.rename_map else None
        policy = GrootRemotePolicy(*parse_groot_url(args.policy), action_horizon=args.action_horizon,
                                   rename_map=rename_map)
        return policy, spec.TIMING.warmup_steps_eval
    from evalkit.lerobot_policy import LeRobotPolicy

    return LeRobotPolicy(args.policy, device=args.device), spec.TIMING.warmup_steps_eval


def run_episodes(env, policy, seed: int, warmup: int, sink, keep_going) -> tuple[int, int]:
    """Episódios seguidos até ``keep_going()`` falhar, o viewer fechar ou Ctrl+C; devolve (sucessos, n)."""
    n = successes = 0
    try:
        while keep_going():
            ep = run_episode(env, policy, seed, warmup, frames=sink)
            n, successes = n + 1, successes + ep["success"]
            result = "sucesso" if ep["success"] else f"falha ({ep['failure']})"
            line = f"semente {seed}: {result}, fase máxima {ep['max_phase']}, {ep['steps']} steps"
            print(f"{line}  [{successes}/{n}]", flush=True)
            sink.episode_done(f"{line}  [{successes}/{n}]")
            seed += 1
    except (ViewerClosed, KeyboardInterrupt):
        # no Ctrl+C o uv repassa o sinal e o processo recebe dois: o segundo cairia no meio do fechamento
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    return successes, n


def main() -> None:
    ap = argparse.ArgumentParser(description="Assiste a uma política no Vials-Lite, episódio após episódio.")
    ap.add_argument("--policy", required=True,
                    help="pasta ou repo do Hub de uma política LeRobot, groot://host:porta ou scripted")
    ap.add_argument("--variant", default="eval", choices=VARIANTS)
    ap.add_argument("--seed", type=int, default=None, help="semente do primeiro episódio (padrão: sorteada)")
    ap.add_argument("--speed", type=float, default=1.0, help="velocidade em relação ao tempo real")
    ap.add_argument("--camera", default="external_D455", help="câmera inicial: external_D455, ego ou free")
    ap.add_argument("--stream", type=int, default=None, metavar="PORTA",
                    help="sem janela: serve as imagens em http://localhost:PORTA (por túnel SSH)")
    ap.add_argument("--action-horizon", type=int, default=None, help="só com groot://")
    ap.add_argument("--rename-map", default=None, help="só com groot://; ver evalkit.runner")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.stream is None and os.environ.get("MUJOCO_GL", "glfw") != "glfw":
        ap.error(f"o viewer precisa de monitor e MUJOCO_GL={os.environ['MUJOCO_GL']} não abre janela; "
                 "tire a variável, ou use --stream PORTA para ver pelo navegador")
    if args.stream is not None and sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")  # antes de importar o mujoco: sem janela
    if args.stream is not None:
        try:
            import PIL  # noqa: F401
        except ImportError:
            ap.error("--stream codifica JPEG com o Pillow: instale o extra lerobot "
                     "ou rode com uv run --with pillow")

    env = VialsLiteEnv(variant=args.variant)
    policy, warmup = load_policy(args, ap, env)
    print(f"render: {env.gl_renderer()}")
    seed = random.randrange(1_000_000) if args.seed is None else args.seed

    if args.stream is not None:
        try:
            sink = StreamSink(env, args.speed, args.camera)
        except ValueError as e:
            ap.error(str(e))
        server = serve(sink, args.stream)
        print(f"abra http://localhost:{args.stream} (de outra máquina: ssh -L {args.stream}:localhost:"
              f"{args.stream} <esta máquina>); Ctrl+C encerra", flush=True)
        try:
            successes, n = run_episodes(env, policy, seed, warmup, sink, lambda: True)
        finally:
            server.shutdown()
            sink.close()
            env.close()
    else:
        import mujoco
        import mujoco.viewer

        sync = ViewerSync(args.speed)
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=sync.key) as viewer:
            sync.viewer = viewer
            if args.camera != "free":
                cam = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
                if cam < 0:
                    ap.error(f"câmera {args.camera!r} não existe (use external_D455, ego ou free)")
                with viewer.lock():
                    viewer.cam.type, viewer.cam.fixedcamid = mujoco.mjtCamera.mjCAMERA_FIXED, cam
            try:
                successes, n = run_episodes(env, policy, seed, warmup, sync, viewer.is_running)
            finally:
                env.close()
        # o viewer roda numa thread própria; se o Python terminar o GLFW com ela ainda aberta, o processo
        # cai com segfault na saída
        for t in threading.enumerate():
            if t is not threading.current_thread() and t.daemon:
                t.join(timeout=5)
    if n:
        print(f"{successes}/{n} episódios completos com sucesso")


if __name__ == "__main__":
    main()
