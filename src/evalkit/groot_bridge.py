"""Ponte para o avaliador do workshop: política no protocolo do servidor GR00T.

O ``scripts/lerobot_eval.py`` do workshop não carrega políticas: ele consulta um servidor de inferência
GR00T por ZMQ (REQ/REP, mensagens em msgpack com os arrays em ``.npy``; ``gr00t_client/server_client.py``)
pelo ``GR00TRemotePolicy`` (``utils/lerobot_interface.py``). Este módulo fala o mesmo protocolo dos dois
lados:

- ``PolicyServer`` + ``ChunkBridge``: serve uma política do LeRobot (``LeRobotPolicy``) ao cliente do
  workshop, que roda no Isaac sem mudanças. É assim que as políticas treinadas aqui são avaliadas na
  cena original.
- ``GrootRemotePolicy``: porta do cliente do workshop para o runner (``--policy groot://host:porta``).
  Confere a ponte no Vials-Lite (o mesmo ACT, direto e pelo servidor, dá os mesmos episódios) e serve
  para consultar um servidor GR00T de verdade.

Protocolo do cliente do workshop (ce807d9):
    pedido    {"endpoint": "get_action", "data": {"observation": obs, "options": None}}
    obs       "video": {câmera: uint8 (1, 1, H, W, 3)};
              "state": {"single_arm": (1, 1, 5), "gripper": (1, 1, 1)}, no espaço do LeRobot;
              "language": {"annotation.human.task_description": [[texto]]}
    resposta  [ação, info], com ação = {"single_arm": (1, T, 5), "gripper": (1, T, 1)} no espaço do LeRobot
    reset     {"endpoint": "reset", "data": {"options": None}}, no início de cada episódio

O cliente executa ``min(T, --action_horizon)`` ações, uma por step, e só consulta de novo quando a fila
acaba. O servidor devolve ``predict_action_chunk[:, :n_action_steps]``, exatamente o que o
``select_action`` do LeRobot 0.6.1 põe na fila; com ``--action_horizon`` ≥ ``n_action_steps`` a execução é
a mesma do runner (100 serve para os dois: o ACT usa 100 e o pi0.5, 50). O padrão do cliente é 16, que mudaria
o protocolo sem erro nenhum: por isso o servidor imprime as consultas por episódio. Com 100, um episódio
de 450 steps que falha tem 5 consultas no ACT e 9 no pi0.5; com 16, teria 28.

Com ``--replay <variante>.rec`` no lugar de ``--policy``, a ponte devolve as ações de uma avaliação gravada
pelo ``scripts/isaac_recorder.py`` (``RecordedActions``): o Isaac repete em malha aberta o que já fez, e o
piso do Isaac contra ele mesmo sai na mesma métrica do replay no Vials-Lite.

Uso (no ambiente do Vials-Lite; o cliente roda no ambiente do Isaac, com o mesmo ``--policy_port``):
    uv run python -m evalkit.groot_bridge --policy <pasta pretrained_model ou repo do Hub> --log ponte.jsonl
    uv run python -m evalkit.groot_bridge --replay <pasta do gravador>/Eval.rec --log ponte.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import signal
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

LANGUAGE_KEY = "annotation.human.task_description"
DEFAULT_PORT = 5555
WARMUP_STEPS = 10  # o lerobot_eval.py do workshop manda a pose inicial nos 10 primeiros steps, sem consultar


# --- formato do fio (o MsgSerializer do workshop, sem a ModalityConfig, que não usamos) ---


def _encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode(obj: dict) -> Any:
    if "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def pack(data: Any) -> bytes:
    import msgpack

    return msgpack.packb(data, default=_encode)


def unpack(data: bytes) -> Any:
    import msgpack

    return msgpack.unpackb(data, object_hook=_decode)


# --- observação e ação: Vials-Lite ↔ GR00T ---


def to_groot_observation(agent_pos: np.ndarray, pixels: dict[str, np.ndarray], instruction: str) -> dict:
    """Observação no formato que o ``GR00TRemotePolicy`` envia, com as dimensões (B=1, T=1)."""
    state = np.asarray(agent_pos, np.float32)
    return {
        "video": {cam: np.asarray(img)[None, None] for cam, img in pixels.items()},
        "state": {"single_arm": state[None, None, :5], "gripper": state[None, None, 5:6]},
        "language": {LANGUAGE_KEY: [[instruction]]},
    }


def from_groot_observation(obs: dict) -> tuple[dict[str, Any], str]:
    """O inverso: (observação no formato do ``VialsLiteEnv``, texto da tarefa)."""
    state = np.concatenate([np.asarray(obs["state"]["single_arm"], np.float32).reshape(-1),
                            np.asarray(obs["state"]["gripper"], np.float32).reshape(-1)])
    if state.shape != (6,):
        raise ValueError(f"estado com {state.size} valores; esperados 5 (single_arm) + 1 (gripper)")
    pixels = {}
    for cam, img in obs.get("video", {}).items():
        img = np.asarray(img)
        if img.ndim < 3 or img.shape[-1] != 3 or int(np.prod(img.shape[:-3])) != 1 or img.dtype != np.uint8:
            raise ValueError(f"câmera {cam!r}: esperado uint8 (1, 1, H, W, 3), veio {img.dtype} {img.shape}")
        pixels[cam] = img.reshape(img.shape[-3:])
    text = obs.get("language", {}).get(LANGUAGE_KEY, "")
    while isinstance(text, (list, tuple)):
        text = text[0] if text else ""
    return {"agent_pos": state, "pixels": pixels}, str(text)


def to_groot_action(chunk: np.ndarray) -> dict[str, np.ndarray]:
    """Chunk (T, 6) no espaço do LeRobot → ação do GR00T, com a dimensão do batch."""
    chunk = np.asarray(chunk, np.float32)
    return {"single_arm": chunk[None, :, :5], "gripper": chunk[None, :, 5:6]}


def from_groot_action(action: dict, horizon: int) -> np.ndarray:
    """O inverso, cortado em ``horizon`` ações como no ``_decode_action_chunk`` do workshop. Shape (T, 6)."""
    full = np.concatenate([np.asarray(action["single_arm"])[0], np.asarray(action["gripper"])[0]], axis=1)
    return full[:horizon].astype(np.float32)


# --- servidor ---


class ChunkBridge:
    """Endpoints ``get_action`` e ``reset`` do servidor GR00T sobre uma política com ``predict_chunk``.

    Conta as consultas por episódio (entre dois ``reset``) e as imprime; com ``log_file``, grava uma linha
    JSON por episódio. É a conferência do ``--action_horizon`` do cliente (ver a docstring do módulo).
    """

    def __init__(self, policy, n_action_steps: int | None = None, log=print, log_file: Path | None = None):
        cfg = getattr(policy, "config", None)
        self.policy = policy
        self.n_action_steps = n_action_steps or cfg.n_action_steps
        features = getattr(cfg, "input_features", None) or {}
        self.cameras = sorted(k.removeprefix("observation.images.") for k in features
                              if k.startswith("observation.images."))
        self._log = log
        self._log_file = log_file
        self.episode = 0  # episódios com pelo menos uma consulta
        self._queries = 0
        self._actions = 0
        self._instructions: set[str] = set()
        self._seen: tuple | None = None  # (formatos das câmeras, texto) da última observação impressa

    def reset(self, options: dict | None = None) -> dict:
        self._close_episode()
        self.policy.reset()
        return {}

    def get_action(self, observation: dict, options: dict | None = None) -> list:
        obs, instruction = from_groot_observation(observation)
        if self.cameras and sorted(obs["pixels"]) != self.cameras:
            raise ValueError(f"câmeras recebidas {sorted(obs['pixels'])}, a política espera {self.cameras} "
                             "(use --rename_map no cliente)")
        seen = (sorted((cam, img.shape) for cam, img in obs["pixels"].items()), instruction)
        if seen != self._seen:  # imprime a primeira observação e cada mudança de formato ou de texto
            self._log(f"[ponte] observação: câmeras {dict(seen[0])}, "
                      f"estado {[round(float(x), 1) for x in obs['agent_pos']]}, texto {instruction!r}")
            self._seen = seen
        chunk = np.asarray(self.policy.predict_chunk(obs, instruction))[: self.n_action_steps]
        self._queries += 1
        self._actions += len(chunk)
        self._instructions.add(instruction)
        return [to_groot_action(chunk), {}]

    def get_modality_config(self) -> dict:
        return {}

    def _close_episode(self) -> None:
        if self._queries == 0:
            return
        self.episode += 1
        record = {"episode": self.episode, "queries": self._queries, "actions": self._actions,
                  "instructions": sorted(self._instructions)}
        extra = self.policy.episode_info() if hasattr(self.policy, "episode_info") else {}
        record.update(extra)
        self._log(f"[ponte] episódio {self.episode}: {self._queries} consultas, até {self._actions} ações "
                  f"(chunks de {self.n_action_steps})" + "".join(f"; {k} {v}" for k, v in extra.items()))
        if self._log_file is not None:
            with self._log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._queries = self._actions = 0
        self._instructions = set()

    def close(self) -> None:
        self._close_episode()


def action_noise(n_steps: int, episode: int, sigma_deg: float) -> np.ndarray:
    """Ruído gaussiano (radianos, ``(n_steps, 6)``) somado às ações gravadas do episódio, semeado pelo número
    do episódio: o mesmo no Isaac (``RecordedActions``) e no replay do Vials-Lite, para medir os dois pisos
    com a mesma perturbação."""
    return np.radians(np.random.default_rng(episode).normal(0.0, sigma_deg, (n_steps, 6)))


class RecordedActions:
    """Ações de uma avaliação gravada no Isaac (``scripts/isaac_recorder.py``), servidas de novo.

    É o replay em malha aberta do lado do Isaac: com a mesma ``--seed`` e o mesmo
    ``--num_episodes``, o Isaac sorteia as mesmas cenas (verificado repetindo uma avaliação), e o
    episódio *k* do cliente recebe as ações do ``ep_k`` do registro. Como o ``ChunkBridge`` chama ``reset``
    também antes do primeiro episódio e depois do último, o episódio só avança num ``reset`` depois de pelo
    menos uma consulta.

    O cliente consulta no step ``WARMUP_STEPS`` e depois a cada ``n_action_steps`` (com ``--action_horizon``
    igual a ``n_action_steps``); cada consulta devolve as ações seguintes do registro, e depois do fim a
    última, como o ``IsaacReplayPolicy`` do runner. Conferência do alinhamento: em cada consulta, as juntas
    recebidas contra as gravadas no step anterior (°), em ``episode_info`` (vai para o ``.ponte.jsonl``).

    Com as mesmas ações, o PhysX repete a gravação (os 10 steps do aquecimento saem idênticos entre duas
    execuções, mesmo em GPUs diferentes), e o replay exato daria um piso trivial. ``action_noise_deg`` soma
    às ações o ``action_noise`` do episódio, a perturbação mínima com que se mede o piso.
    """

    def __init__(self, rec_dir: Path, n_action_steps: int = 100, action_noise_deg: float = 0.0):
        from vials_lite.joint_mapping import lerobot_to_radians, radians_to_lerobot

        from .runner import isaac_joint_order

        self.rec_dir = Path(rec_dir)
        names = json.loads((self.rec_dir / "meta.json").read_text(encoding="utf-8"))["joint_names"]
        self._order = isaac_joint_order(names)
        self._to_lerobot, self._to_radians = radians_to_lerobot, lerobot_to_radians
        self.name = f"replay-isaac:{self.rec_dir.parent.name}/{self.rec_dir.name}"
        self.config = SimpleNamespace(n_action_steps=n_action_steps, input_features={})
        self.action_noise_deg = action_noise_deg
        self.episode = -1
        self._served = True  # o primeiro reset abre o episódio 0
        self._actions: np.ndarray | None = None
        self._joints: np.ndarray | None = None
        self._k = WARMUP_STEPS
        self._dev: list[float] = []

    def reset(self) -> None:
        if self._served:
            self.episode += 1
            self._actions = None  # carregado na 1ª consulta: o reset depois do último episódio não lê nada
        self._served = False
        self._k = WARMUP_STEPS
        self._dev = []

    def predict_chunk(self, observation: dict, instruction: str) -> np.ndarray:
        if self._actions is None:
            ep = np.load(self.rec_dir / f"ep_{self.episode:04d}.npz")
            rad = ep["action"][:, self._order]
            if self.action_noise_deg:
                rad = rad + action_noise(len(rad), self.episode, self.action_noise_deg)
            self._actions = self._to_lerobot(rad).astype(np.float32)
            self._joints = ep["joint_pos"][:, self._order]
        got = self._to_radians(np.asarray(observation["agent_pos"], np.float64))
        ref = self._joints[min(self._k - 1, len(self._joints) - 1)]
        self._dev.append(float(np.degrees(np.abs(got - ref)).max()))
        n = self.config.n_action_steps
        idx = np.minimum(np.arange(self._k, self._k + n), len(self._actions) - 1)
        self._k += n
        self._served = True
        return self._actions[idx]

    def episode_info(self) -> dict:
        """Episódio do registro e desvio das juntas (°) na 1ª consulta e no pior caso."""
        if not self._dev:
            return {}
        return {"rec_episode": self.episode, "dev_first_deg": round(self._dev[0], 3),
                "dev_max_deg": round(max(self._dev), 3)}


class PolicyServer:
    """Servidor ZMQ REP compatível com o ``PolicyClient`` do workshop (``ping``, ``reset``, ``get_action``,
    ``get_modality_config``, ``kill``). Erros voltam como ``{"error": ...}``, que o cliente levanta.

    ``port=0`` escolhe uma porta livre (testes); a porta usada fica em ``self.port``.
    """

    def __init__(self, bridge: ChunkBridge, host: str = "127.0.0.1", port: int = DEFAULT_PORT):
        import zmq

        self.bridge = bridge
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        if port == 0:
            self.port = self.socket.bind_to_random_port(f"tcp://{host}")
        else:
            self.socket.bind(f"tcp://{host}:{port}")
            self.port = port
        self.running = True
        self.endpoints = {  # nome → (função, recebe ``data``)
            "ping": (lambda: {"status": "ok", "message": "Server is running"}, False),
            "kill": (self._kill, False),
            "get_action": (bridge.get_action, True),
            "reset": (bridge.reset, True),
            "get_modality_config": (bridge.get_modality_config, False),
        }

    def _kill(self) -> None:
        self.running = False

    def handle(self, message: bytes) -> bytes:
        try:
            request = unpack(message)
            endpoint = request.get("endpoint", "get_action")
            if endpoint not in self.endpoints:
                raise ValueError(f"endpoint desconhecido: {endpoint!r}")
            handler, takes_data = self.endpoints[endpoint]
            result = handler(**(request.get("data") or {})) if takes_data else handler()
            return pack(result)
        except Exception as e:  # o REP precisa responder sempre; o cliente levanta o erro
            traceback.print_exc()
            return pack({"error": f"{type(e).__name__}: {e}"})

    def serve(self) -> None:
        while self.running:
            self.socket.send(self.handle(self.socket.recv()))

    def close(self) -> None:
        self.bridge.close()
        self.socket.close(linger=0)
        self.context.term()


# --- cliente (porta do GR00TRemotePolicy do workshop) ---


class GrootRemotePolicy:
    """Política do runner que consulta um servidor no protocolo do GR00T: uma ação por step, com fila.

    Como no ``GR00TRemotePolicy`` do workshop, só consulta o servidor quando a fila acaba e usa as primeiras
    ``action_horizon`` ações do chunk. O ``reset`` limpa a fila e reinicia o estado do servidor.
    ``rename_map`` troca o nome das câmeras antes de enviar, como o ``--rename_map`` do cliente do workshop
    (o GR00T do curso espera ``front`` e ``wrist``).
    """

    def __init__(self, host: str = "localhost", port: int = DEFAULT_PORT, action_horizon: int = 16,
                 timeout_ms: int = 300_000, rename_map: dict[str, str] | None = None):
        import zmq

        self.name = f"groot:{host}:{port}"
        self.action_horizon = action_horizon
        self.rename_map = rename_map or {}
        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{host}:{port}")
        self._queue: deque[np.ndarray] = deque()
        self.queries = 0
        self.call("ping", requires_input=False)

    def call(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        self._socket.send(pack(request))
        try:
            response = unpack(self._socket.recv())
        except self._zmq.error.Again as e:
            raise TimeoutError(f"o servidor {self.name} não respondeu a {endpoint!r}") from e
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"erro no servidor: {response['error']}")
        return response

    def reset(self) -> None:
        self.call("reset", {"options": None})
        self._queue.clear()

    def act(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        if not self._queue:
            pixels = {self.rename_map.get(cam, cam): img
                      for cam, img in observation.get("pixels", {}).items()}
            obs = to_groot_observation(observation["agent_pos"], pixels, instruction)
            action, _info = self.call("get_action", {"observation": obs, "options": None})
            self._queue.extend(from_groot_action(action, self.action_horizon))
            self.queries += 1
        return self._queue.popleft()

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()


def parse_groot_url(url: str) -> tuple[str, int] | None:
    """``groot://host:porta`` → (host, porta); outra coisa → None."""
    if not url.startswith("groot://"):
        return None
    host, _, port = url.removeprefix("groot://").rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"use groot://host:porta, veio {url!r}")
    return host, int(port)


def _interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Serve uma política do LeRobot no protocolo do servidor GR00T, para o lerobot_eval.py "
                    "do workshop. No cliente, use --action_horizon 100.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--policy", help="pasta ou repo do Hub de uma política LeRobot")
    src.add_argument("--replay", type=Path,
                     help="pasta <variante>.rec do gravador: devolve as ações gravadas (malha aberta)")
    ap.add_argument("--action-steps", type=int, default=100,
                    help="com --replay: ações por consulta (= --action_horizon do cliente)")
    ap.add_argument("--action-noise-deg", type=float, default=0.0,
                    help="com --replay: desvio-padrão (°) do ruído somado às ações gravadas (action_noise)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="interface de escuta (padrão: só a máquina local; '*' abre para a rede)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log", type=Path, default=None, help="JSONL com as consultas de cada episódio")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # as linhas por episódio aparecem na hora, mesmo com | tee
    # SIGTERM (kill, scripts/isaac_eval_cells.sh) encerra como o Ctrl-C: fecha o log do último episódio
    signal.signal(signal.SIGTERM, _interrupt)

    t0 = time.perf_counter()
    if args.replay:
        policy = RecordedActions(args.replay, n_action_steps=args.action_steps,
                                 action_noise_deg=args.action_noise_deg)
    else:
        from .lerobot_policy import LeRobotPolicy

        policy = LeRobotPolicy(args.policy, device=args.device)
    bridge = ChunkBridge(policy, log_file=args.log)
    server = PolicyServer(bridge, host=args.host, port=args.port)
    print(f"[ponte] {policy.name} carregada em {time.perf_counter() - t0:.0f} s; chunks de "
          f"{bridge.n_action_steps} ações; câmeras {bridge.cameras}; ouvindo em {args.host}:{server.port}")
    horizon = max(100, bridge.n_action_steps)
    print(f"[ponte] no cliente: --policy_port {server.port} --action_horizon {horizon}")
    try:
        server.serve()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        print(f"[ponte] encerrada depois de {bridge.episode} episódios")


if __name__ == "__main__":
    main()
