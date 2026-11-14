"""Assiste a uma política no Vials-Lite: episódios consecutivos no viewer do MuJoCo, até fechar a janela.

Cada episódio segue o protocolo do ``evalkit.runner`` (aquecimento, 450 steps, término no sucesso), com
as sementes ``seed``, ``seed + 1``, ... A cada episódio o terminal mostra a semente, o resultado, a
fase máxima e a categoria de falha. Para repetir um episódio visto aqui com o relatório completo:
``python -m evalkit.runner --policy <política> --seed <semente> --episodes 1 --video 1``.

Não grava nada: é para ver. Para medir, use o runner.

``--policy scripted`` usa a demo roteirizada (``vials_lite.scripted``), sem LeRobot nem política
treinada. Sem ``--seed``, a primeira semente é sorteada.

O viewer abre na câmera externa (a que a política vê). No viewer: ``[`` e ``]`` trocam de câmera
(``ego`` é a do punho), Esc passa para a câmera livre (arrastar gira, botão direito desloca, roda
aproxima) e a barra de espaço pausa. Fechar a janela ou Ctrl+C encerra.

Precisa de monitor: o viewer não roda com ``MUJOCO_GL=egl`` (por SSH, use o ``--video`` do runner).

Uso::

    uv run python scripts/watch_policy.py --policy scripted
    uv run python scripts/watch_policy.py --policy arjunjuneja/act_vials_50k --variant dr-eval
    uv run python scripts/watch_policy.py --policy groot://localhost:5555 --action-horizon 16 \\
        --rename-map '{"external_D455": "front", "ego": "wrist"}'
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

from evalkit.runner import run_episode
from vials_lite import spec
from vials_lite.env import VARIANTS, VialsLiteEnv
from vials_lite.scripted import ScriptedPickPlace


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


class ViewerSync:
    """Recebe um frame por step do ``run_episode`` (o parâmetro ``frames``), atualiza o viewer e
    segura o ritmo em ``speed`` × tempo real; a barra de espaço pausa. O frame em si é descartado."""

    def __init__(self, speed: float) -> None:
        self.viewer = None
        self.paused = False
        self.dt = 1.0 / (spec.TIMING.control_hz * speed)
        self._next = time.perf_counter()

    def key(self, keycode: int) -> None:
        if keycode == ord(" "):
            self.paused = not self.paused

    def append(self, _frame) -> None:
        if not self.viewer.is_running():
            raise ViewerClosed
        self.viewer.sync()
        while self.paused and self.viewer.is_running():
            time.sleep(0.05)
            self.viewer.sync()
            self._next = time.perf_counter()
        self._next += self.dt
        delay = self._next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:  # política mais lenta que o tempo real: não acumula atraso
            self._next = time.perf_counter()


def main() -> None:
    ap = argparse.ArgumentParser(description="Assiste a uma política no Vials-Lite, episódio após episódio.")
    ap.add_argument("--policy", required=True,
                    help="pasta ou repo do Hub de uma política LeRobot, groot://host:porta ou scripted")
    ap.add_argument("--variant", default="eval", choices=VARIANTS)
    ap.add_argument("--seed", type=int, default=None, help="semente do primeiro episódio (padrão: sorteada)")
    ap.add_argument("--speed", type=float, default=1.0, help="velocidade em relação ao tempo real")
    ap.add_argument("--camera", default="external_D455", help="câmera inicial do viewer, ou free")
    ap.add_argument("--action-horizon", type=int, default=None, help="só com groot://")
    ap.add_argument("--rename-map", default=None, help="só com groot://; ver evalkit.runner")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if os.environ.get("MUJOCO_GL", "glfw") != "glfw":
        ap.error(f"o viewer precisa de monitor e MUJOCO_GL={os.environ['MUJOCO_GL']} não abre janela; "
                 "tire a variável (ou use MUJOCO_GL=glfw)")

    env = VialsLiteEnv(variant=args.variant)
    if args.policy == "scripted":
        policy, warmup = ScriptedPolicy(env), 0
    elif args.policy.startswith("groot://"):
        from evalkit.groot_bridge import GrootRemotePolicy, parse_groot_url

        if args.action_horizon is None:
            ap.error("--policy groot://... exige --action-horizon")
        rename_map = json.loads(args.rename_map) if args.rename_map else None
        policy = GrootRemotePolicy(*parse_groot_url(args.policy), action_horizon=args.action_horizon,
                                   rename_map=rename_map)
        warmup = spec.TIMING.warmup_steps_eval
    else:
        from evalkit.lerobot_policy import LeRobotPolicy

        policy, warmup = LeRobotPolicy(args.policy, device=args.device), spec.TIMING.warmup_steps_eval
    print(f"render: {env.gl_renderer()}")

    seed = random.randrange(1_000_000) if args.seed is None else args.seed
    n = successes = 0
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
            while viewer.is_running():
                ep = run_episode(env, policy, seed, warmup, frames=sync)
                n, successes = n + 1, successes + ep["success"]
                result = "sucesso" if ep["success"] else f"falha ({ep['failure']})"
                print(f"semente {seed}: {result}, fase máxima {ep['max_phase']}, {ep['steps']} steps"
                      f"  [{successes}/{n}]")
                seed += 1
        except (ViewerClosed, KeyboardInterrupt):
            pass
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
