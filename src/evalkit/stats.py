"""Estatística para taxas de sucesso de políticas.

Implementado:
    wilson_interval       — IC de Wilson para uma proporção
    newcombe_difference   — IC da diferença entre duas proporções independentes (Newcombe, 1998,
                            método 10: híbrido de escores de Wilson)
    paired_difference     — IC da diferença entre duas proporções pareadas (Newcombe, 1998b,
                            método 10: o mesmo híbrido, com a correlação dos pares)
    mcnemar_exact         — p bilateral do teste exato de McNemar (binomial nos pares discordantes)

Não implementado: comparação sequencial com parada antecipada, como no STEP
(Snyder et al., RSS 2025, https://arxiv.org/abs/2503.10966).

Exemplo: 11 sucessos em 20 episódios → IC 95% ≈ [34%, 74%]; com 20 episódios, o intervalo ainda é
largo demais para separar políticas próximas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{100 * self.estimate:.1f}% [{100 * self.low:.1f}%, {100 * self.high:.1f}%]"


def wilson_interval(successes: int, n: int, z: float = Z_95) -> Interval:
    if n <= 0:
        raise ValueError("n deve ser positivo")
    if not 0 <= successes <= n:
        raise ValueError("successes fora de [0, n]")
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return Interval(p, max(0.0, center - half), min(1.0, center + half))


def newcombe_difference(s1: int, n1: int, s2: int, n2: int, z: float = Z_95) -> Interval:
    """IC para p1 − p2. Se o intervalo não contém 0, a diferença é significativa ao nível z."""
    a = wilson_interval(s1, n1, z)
    b = wilson_interval(s2, n2, z)
    d = a.estimate - b.estimate
    low = d - math.sqrt((a.estimate - a.low) ** 2 + (b.high - b.estimate) ** 2)
    high = d + math.sqrt((a.high - a.estimate) ** 2 + (b.estimate - b.low) ** 2)
    return Interval(d, low, high)


def paired_difference(both: int, only1: int, only2: int, neither: int, z: float = Z_95) -> Interval:
    """IC para p1 − p2 com os mesmos episódios nos dois lados (Newcombe, Stat Med 1998, 17:2635).

    Recebe a tabela 2×2 dos pares: sucesso nos dois, só no 1, só no 2, em nenhum. Combina os
    intervalos de Wilson das duas margens como o ``newcombe_difference``, descontando a correlação
    φ entre os pares (com a correção de continuidade do artigo quando φ > 0).
    """
    n = both + only1 + only2 + neither
    if n <= 0:
        raise ValueError("n deve ser positivo")
    if min(both, only1, only2, neither) < 0:
        raise ValueError("contagens negativas")
    a = wilson_interval(both + only1, n, z)
    b = wilson_interval(both + only2, n, z)
    margins = (both + only1) * (only2 + neither) * (both + only2) * (only1 + neither)
    phi = 0.0
    if margins > 0:
        num = both * neither - only1 * only2
        if num > 0:
            num = max(num - n / 2, 0.0)
        phi = num / math.sqrt(margins)
    d = a.estimate - b.estimate
    dl, du = a.estimate - a.low, b.high - b.estimate
    el, eu = a.high - a.estimate, b.estimate - b.low
    low = d - math.sqrt(max(0.0, dl * dl - 2 * phi * dl * du + du * du))
    high = d + math.sqrt(max(0.0, el * el - 2 * phi * el * eu + eu * eu))
    return Interval(d, max(-1.0, low), min(1.0, high))


def mcnemar_exact(only1: int, only2: int) -> float:
    """p bilateral do McNemar exato: binomial(only1 + only2, 1/2) nos pares discordantes."""
    n = only1 + only2
    if n == 0:
        return 1.0
    k = min(only1, only2)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
