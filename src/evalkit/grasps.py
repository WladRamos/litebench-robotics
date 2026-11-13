"""Fechamentos da garra: onde a garra fecha em relação ao frasco, se ele sobe e se a garra o segue.

"Tocou mas não levantou" (``taxonomy``) junta dois casos que pedem correções diferentes: a garra
fecha **ao lado** do frasco e só esbarra nele (o contato passa de 2 N, mas não há pega), ou fecha
**sobre** ele e o frasco escorrega. Medido nos episódios do dataset e nos rollouts do ACT, os
fechamentos que levantam têm o eixo do frasco a ~9–11 mm do site da garra; os que falham, a 30–65 mm.

Cada fechamento é um step em que o comando da garra cruza ``CLOSE_COMMAND`` para baixo. Para ele,
guarda-se o frasco cujo eixo está mais perto do site, essa distância, se o frasco sobe mais de
``LIFT_MIN_RISE_M`` nos ``LIFT_WINDOW_STEPS`` seguintes, e as posições xy do site e do frasco no
início do episódio. Do 1º fechamento de cada episódio sai **quanto a garra segue o frasco**: a
inclinação da regressão linear da posição do site contra a do frasco, por eixo (1 = vai onde o
frasco está; 0 = vai sempre ao mesmo lugar). Operadores: 0,94 em y; ACT na ``eval``: 0,18.

NumPy puro: lê os arrays por step do runner (``steps/ep_XXX.npz``) e as linhas de ``episodes.jsonl``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from .progress import distance_to_vial_axis

# Comando da garra (LeRobot, 0 fechada..100 aberta) que, cruzado para baixo, marca um fechamento.
# Não calibrado.
CLOSE_COMMAND = 20.0
LIFT_WINDOW_STEPS = 120  # 2 s a 60 Hz
LIFT_MIN_RISE_M = 0.03
# Fechou "sobre o frasco": eixo a menos de 2 cm do site. Os que levantam ficam a ~9–11 mm e os que
# falham, a 30–65 mm. Limiar heurístico, não calibrado
NEAR_AXIS_M = 0.02
MIN_EPISODES_FOR_SLOPE = 5


@dataclass(frozen=True)
class Closure:
    step: int
    vial: int  # frasco com o eixo mais perto do site no instante do fechamento
    axis_dist_mm: float  # distância do site ao segmento do eixo desse frasco
    lifted: bool  # o frasco subiu > LIFT_MIN_RISE_M nos LIFT_WINDOW_STEPS seguintes
    site_xy: tuple[float, float]  # site da garra no fechamento (m)
    vial_xy0: tuple[float, float]  # origem desse frasco no 1º step do episódio (m)


def find_closures(
    gripper_command: np.ndarray,
    gripper_pos: np.ndarray,
    vial_pos: np.ndarray,
    vial_quat: np.ndarray,
    ignore_vial: int | Sequence[int] | None = None,
) -> list[Closure]:
    """Fechamentos de um episódio. Arrays por step: comando da garra (T,), site (T, 3), frascos
    (T, V, 3) e (T, V, 4). ``ignore_vial``: frascos que já começam no rack (não são alvo)."""
    cmd = np.asarray(gripper_command, float)
    ignored = set() if ignore_vial is None else {int(i) for i in np.atleast_1d(ignore_vial)}
    candidates = [v for v in range(vial_pos.shape[1]) if v not in ignored]
    if not candidates:
        return []
    out = []
    for j in np.flatnonzero((cmd[:-1] > CLOSE_COMMAND) & (cmd[1:] <= CLOSE_COMMAND)) + 1:
        dists = [distance_to_vial_axis(gripper_pos[j], vial_pos[j, v], vial_quat[j, v]) for v in candidates]
        v = candidates[int(np.argmin(dists))]
        z = vial_pos[j : j + LIFT_WINDOW_STEPS, v, 2]
        out.append(Closure(
            step=int(j),
            vial=v,
            axis_dist_mm=round(1000 * min(dists), 1),
            lifted=bool(z.max() - z[0] > LIFT_MIN_RISE_M),
            site_xy=tuple(round(float(x), 4) for x in gripper_pos[j, :2]),
            vial_xy0=tuple(round(float(x), 4) for x in vial_pos[0, v, :2]),
        ))
    return out


def closures_from_arrays(arrays: dict[str, np.ndarray], ignore_vial=None) -> list[dict]:
    """``find_closures`` sobre os arrays do runner, em dicionários para o ``episodes.jsonl``."""
    found = find_closures(arrays["action"][:, 5], arrays["gripper_pos"], arrays["vial_pos"],
                          arrays["vial_quat"], ignore_vial)
    return [asdict(c) for c in found]


def tracking_slope(site: np.ndarray, vial: np.ndarray) -> float | None:
    """Inclinação da regressão de ``site`` contra ``vial`` (1D); None se o frasco quase não varia."""
    site, vial = np.asarray(site, float), np.asarray(vial, float)
    if len(vial) < MIN_EPISODES_FOR_SLOPE or vial.std() < 1e-4:
        return None
    return float(np.polyfit(vial, site, 1)[0])


@dataclass(frozen=True)
class ClosureStats:
    n_episodes: int  # episódios com a análise (runs antigos não têm ``closures``)
    n_closures: int
    per_episode_median: float
    lifted: int
    lifted_median_dist_mm: float | None
    not_lifted_median_dist_mm: float | None
    first_lifted: int  # episódios cujo 1º fechamento levantou
    episodes_with_closure: int
    slope_x: float | None
    slope_y: float | None
    # entre os episódios "tocou mas não levantou": nenhum fechamento perto do eixo / algum perto
    never_lifted_closed_beside: int
    never_lifted_closed_on_vial: int


def closure_stats(episodes: list[dict], never_lifted_label: str) -> ClosureStats | None:
    """Agrega os ``closures`` das linhas de ``episodes.jsonl``; None se nenhuma os tiver."""
    eps = [e for e in episodes if "closures" in e]
    if not eps:
        return None
    every = [c for e in eps for c in e["closures"]]
    up = [c["axis_dist_mm"] for c in every if c["lifted"]]
    down = [c["axis_dist_mm"] for c in every if not c["lifted"]]
    firsts = [e["closures"][0] for e in eps if e["closures"]]
    site = np.array([c["site_xy"] for c in firsts]).reshape(-1, 2)
    vial = np.array([c["vial_xy0"] for c in firsts]).reshape(-1, 2)
    beside = on_vial = 0
    for e in eps:
        if e.get("failure") == never_lifted_label:
            if any(c["axis_dist_mm"] <= 1000 * NEAR_AXIS_M for c in e["closures"]):
                on_vial += 1
            else:
                beside += 1
    return ClosureStats(
        n_episodes=len(eps),
        n_closures=len(every),
        per_episode_median=float(np.median([len(e["closures"]) for e in eps])),
        lifted=len(up),
        lifted_median_dist_mm=float(np.median(up)) if up else None,
        not_lifted_median_dist_mm=float(np.median(down)) if down else None,
        first_lifted=sum(c["lifted"] for c in firsts),
        episodes_with_closure=len(firsts),
        slope_x=tracking_slope(site[:, 0], vial[:, 0]),
        slope_y=tracking_slope(site[:, 1], vial[:, 1]),
        never_lifted_closed_beside=beside,
        never_lifted_closed_on_vial=on_vial,
    )
