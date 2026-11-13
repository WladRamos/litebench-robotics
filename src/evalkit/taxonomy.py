"""Taxonomia automática de falhas por regras sobre o resumo do episódio.

As categorias partem da falha típica observada no GR00T N1.6 na cena original do workshop:
"agarra e solta o frasco várias vezes sem encaixar, gastando os 450 steps". As regras são
heurísticas, e os limiares marcados como não calibrados foram escolhidos à mão.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .progress import Phase


class Failure(StrEnum):
    NONE = "sucesso"
    NEVER_REACHED = "nunca alcançou um frasco"
    NEVER_GRASPED = "alcançou mas nunca agarrou"
    NEVER_LIFTED = "tocou mas não levantou"  # contato > 2 N sem tirar o frasco do tapete
    DROPPED_IN_TRANSIT = "derrubou no transporte"
    REGRASP_LOOP = "loop de re-agarre"
    MISSED_SLOT = "soltou em cima do rack, fora dos furos"
    PLACED_TILTED = "colocou inclinado"
    MISSED_RACK = "soltou fora do rack"
    TIMEOUT_NEAR_GOAL = "timeout perto do objetivo"
    OTHER = "outra"


@dataclass(frozen=True)
class EpisodeSummary:
    success: bool
    max_phase: Phase
    n_grasps: int  # transições soltar→agarrar
    n_releases_outside_rack: int
    released_inside_rack_tilted: bool  # solto na caixa do rack e inclinado (na soltura ou no fim)
    min_vial_to_slot_dist_m: float
    released_on_rack_top: bool = False  # solto sobre o rack, sem entrar num furo


REGRASP_LOOP_MIN = 3  # heurístico, não calibrado


def classify(ep: EpisodeSummary) -> Failure:
    if ep.success:
        return Failure.NONE
    if ep.max_phase < Phase.REACHED:
        return Failure.NEVER_REACHED
    if ep.max_phase < Phase.GRASPED:
        return Failure.NEVER_GRASPED
    if ep.max_phase < Phase.LIFTED:
        return Failure.NEVER_LIFTED
    if ep.n_grasps >= REGRASP_LOOP_MIN:
        return Failure.REGRASP_LOOP
    if ep.released_on_rack_top:
        return Failure.MISSED_SLOT
    if ep.released_inside_rack_tilted:
        return Failure.PLACED_TILTED
    if ep.max_phase >= Phase.OVER_RACK and ep.n_releases_outside_rack > 0:
        return Failure.MISSED_RACK
    if ep.max_phase < Phase.OVER_RACK and ep.n_releases_outside_rack > 0:
        return Failure.DROPPED_IN_TRANSIT
    if ep.max_phase >= Phase.OVER_RACK:
        return Failure.TIMEOUT_NEAR_GOAL
    return Failure.OTHER
