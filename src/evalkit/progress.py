"""Progresso da tarefa a partir de estado privilegiado (verdade de solo).

A ideia é trocar o "deu certo / não deu" por **até onde o episódio chegou**. As fases são
ordinais; o progresso de um episódio é a maior fase atingida, e há também uma versão contínua
(dentro da fase, distância normalizada até o próximo marco).

Por vir de verdade de solo, serve também de referência para estimadores de progresso que só
veem o vídeo (reward models como Robometer e TOPReward).

``ProgressTracker`` consome um ``ProgressState`` por step (o que o ``Probe`` lê do simulador) e
acumula a fase máxima e os números que a taxonomia usa (``EpisodeSummary``). Os limiares reusam
os do critério de sucesso (``spec.SUCCESS``); os que não vêm do workshop são heurísticos e estão
marcados como não calibrados.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from vials_lite import spec
from vials_lite.success import quat_inverse, quat_rotate, vial_geometry_ok


class Phase(IntEnum):
    START = 0
    REACHED = 1  # garra a < REACH_DISTANCE_M do eixo de algum frasco
    GRASPED = 2  # contato garra↔frasco > 2 N
    LIFTED = 3  # frasco agarrado acima de 5,5 cm (mesmo limiar do evento [GRASP] do workshop)
    OVER_RACK = 4  # frasco agarrado com projeção xy dentro da caixa do rack
    INSERTED = 5  # frasco já agarrado, agora solto, dentro da caixa do rack (z local < 0,1)
    UPRIGHT = 6  # INSERTED + vertical (> 0.7)
    SUCCESS = 7  # UPRIGHT confirmado por 25 steps (critério oficial)


N_PHASES = len(Phase)
REACH_DISTANCE_M = 0.03  # heurístico, não calibrado
# Base do frasco (z local do rack) acima da qual a soltura foi sobre o rack e não num furo: assentado
# no furo, a base fica em 0,014; em pé sobre o tampo, em ~0,09. Heurístico, não calibrado
ON_RACK_TOP_MIN_Z = 0.05
RELEASE_DEBOUNCE_STEPS = 3  # contato perdido por 3 steps seguidos = soltou (o contato pisca); não calibrado


def phase_progress(max_phase: Phase) -> float:
    """Progresso discreto em [0, 1]."""
    return int(max_phase) / (N_PHASES - 1)


@dataclass
class ProgressState:
    """Estado privilegiado de um step. Arrays com uma linha por frasco."""

    gripper_pos: np.ndarray  # (3,) site da ponta da garra
    vial_pos: np.ndarray  # (V, 3) origem do frasco (base do corpo)
    vial_quat: np.ndarray  # (V, 4) wxyz
    rack_pos: np.ndarray  # (3,)
    rack_quat: np.ndarray  # (4,) wxyz
    jaw_contact_force: np.ndarray  # (V,) N
    success: bool = False


def distance_to_vial_axis(point: np.ndarray, vial_pos: np.ndarray, vial_quat: np.ndarray) -> float:
    """Distância do ponto ao segmento do eixo do frasco (da ponta do cone ao topo da tampa)."""
    axis = quat_rotate(np.asarray(vial_quat, float), np.array([0.0, 0.0, 1.0]))
    rel = np.asarray(point, float) - vial_pos
    s = np.clip(rel @ axis, spec.VIAL_TIP_Z, spec.VIAL_TOP_Z)
    return float(np.linalg.norm(rel - s * axis))


def _rack_local(p: np.ndarray, rack_pos: np.ndarray, rack_quat: np.ndarray) -> np.ndarray:
    return quat_rotate(quat_inverse(np.asarray(rack_quat, float)), np.asarray(p) - rack_pos)


def _over_rack_box(local: np.ndarray, sp: spec.SuccessParams = spec.SUCCESS) -> bool:
    return bool(
        sp.rack_local_x[0] <= local[0] <= sp.rack_local_x[1]
        and sp.rack_local_y[0] <= local[1] <= sp.rack_local_y[1]
    )


@dataclass
class ProgressTracker:
    """Fase por step e resumo do episódio.

    ``ignore_vial``: frasco que já começa no rack (o reset do workshop põe um com p = 0,33), ou uma
    sequência deles (cenas reconstruídas do dataset podem ter dois). Ele só conta para
    INSERTED/UPRIGHT depois de ser agarrado, como no critério de sucesso.
    """

    n_vials: int = 3
    params: spec.SuccessParams = spec.SUCCESS
    ignore_vial: int | Sequence[int] | None = None
    max_phase: Phase = field(init=False, default=Phase.START)
    phases: list[int] = field(init=False, default_factory=list)
    n_grasps: int = field(init=False, default=0)
    n_releases_outside_rack: int = field(init=False, default=0)
    released_inside_rack_tilted: bool = field(init=False, default=False)
    released_on_rack_top: bool = field(init=False, default=False)
    min_vial_to_slot_dist_m: float = field(init=False, default=float("inf"))
    # algum frasco já agarrado está solto, dentro da caixa do rack e deitado (tombou) agora
    tilted_in_rack_now: bool = field(init=False, default=False)
    _ever_grasped: np.ndarray = field(init=False)
    _carrying: np.ndarray = field(init=False)
    _no_contact: np.ndarray = field(init=False)
    _ignored: set[int] = field(init=False)

    def __post_init__(self) -> None:
        self._ever_grasped = np.zeros(self.n_vials, dtype=bool)
        self._carrying = np.zeros(self.n_vials, dtype=bool)
        self._no_contact = np.zeros(self.n_vials, dtype=int)
        ignore = [] if self.ignore_vial is None else np.atleast_1d(self.ignore_vial)
        self._ignored = {int(i) for i in ignore}

    def update(self, s: ProgressState) -> Phase:
        p = self.params
        slots = np.array(spec.RACK_SLOTS_LOCAL)[:, :2]
        grasped = np.asarray(s.jaw_contact_force) > p.grasp_force_threshold_n
        self._ever_grasped |= grasped
        phase = Phase.START
        self.tilted_in_rack_now = False
        for i in range(self.n_vials):
            pos, quat = s.vial_pos[i], s.vial_quat[i]
            local = _rack_local(pos, s.rack_pos, s.rack_quat)
            vertical, in_box = vial_geometry_ok(pos, quat, s.rack_pos, s.rack_quat, p)
            lifted = bool(grasped[i] and pos[2] > p.grasp_min_height)
            # um agarre é cada vez que o frasco sai do tapete preso na garra; a soltura é o fim disso
            self._no_contact[i] = 0 if grasped[i] else self._no_contact[i] + 1
            if lifted and not self._carrying[i]:
                self._carrying[i] = True
                self.n_grasps += 1
            elif self._carrying[i] and self._no_contact[i] >= RELEASE_DEBOUNCE_STEPS:
                self._carrying[i] = False
                if not in_box:
                    self.n_releases_outside_rack += 1
                elif local[2] > ON_RACK_TOP_MIN_Z:
                    self.released_on_rack_top = True
                elif not vertical:
                    self.released_inside_rack_tilted = True
            if self._ever_grasped[i]:
                d = float(np.min(np.linalg.norm(slots - local[:2], axis=1)))
                self.min_vial_to_slot_dist_m = min(self.min_vial_to_slot_dist_m, d)

            if i in self._ignored and not self._ever_grasped[i]:
                continue
            if distance_to_vial_axis(s.gripper_pos, pos, quat) < REACH_DISTANCE_M:
                phase = max(phase, Phase.REACHED)
            if grasped[i]:
                phase = max(phase, Phase.GRASPED)
                if lifted:
                    phase = max(phase, Phase.LIFTED)
                    if _over_rack_box(local, p):
                        phase = max(phase, Phase.OVER_RACK)
            elif self._ever_grasped[i] and in_box:
                phase = max(phase, Phase.UPRIGHT if vertical else Phase.INSERTED)
                self.tilted_in_rack_now |= not vertical
        if s.success:
            phase = Phase.SUCCESS
        self.max_phase = max(self.max_phase, phase)
        self.phases.append(int(phase))
        return phase

    def summary(self, success: bool):
        from .taxonomy import EpisodeSummary

        return EpisodeSummary(
            success=success,
            max_phase=Phase.SUCCESS if success else self.max_phase,
            n_grasps=self.n_grasps,
            n_releases_outside_rack=self.n_releases_outside_rack,
            released_inside_rack_tilted=self.released_inside_rack_tilted or self.tilted_in_rack_now,
            min_vial_to_slot_dist_m=self.min_vial_to_slot_dist_m,
            released_on_rack_top=self.released_on_rack_top,
        )
