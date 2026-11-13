"""Interfaces do evalkit: o que uma política e um ambiente precisam oferecer.

O evalkit é independente de simulador: Vials-Lite (MuJoCo) é o primeiro backend de ambiente;
outros simuladores e outras tarefas podem entrar, desde que forneçam estado privilegiado num
``Probe``. As métricas de fase (``progress``, ``taxonomy``, ``grasps``), porém, são da tarefa
Vials-to-Rack.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class Policy(Protocol):
    """Política que recebe observações no formato LeRobot e devolve UMA ação (6,) por step.

    Políticas com action chunking guardam a fila internamente (como o ``select_action`` do
    LeRobot e o cliente GR00T do workshop). Políticas que sorteiam na inferência podem oferecer
    ``seed(seed: int)``, que o runner chama antes de cada episódio (``runner.policy_seed``).
    """

    name: str

    def reset(self) -> None: ...

    def act(self, observation: dict[str, Any], instruction: str) -> np.ndarray: ...


class Probe(Protocol):
    """Extrai, do estado do simulador, o que as métricas precisam (poses, contatos, eventos)."""

    def read(self) -> dict[str, Any]: ...


# Backends implementados:
#   LeRobotPolicy (lerobot_policy.py)   — from_pretrained + select_action; testado com ACT e pi0.5
#   ReplayPolicy (runner.py)            — ações gravadas de um episódio do dataset, em malha aberta
#   GrootRemotePolicy (groot_bridge.py) — cliente do servidor GR00T (``--policy groot://host:porta``)
# No Vials-Lite, o papel do Probe é feito pela função runner.probe (info["privileged"] → ProgressState).
