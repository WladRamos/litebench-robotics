"""Isaac × Vials-Lite nas mesmas cenas, episódio a episódio.

Três leituras sobre os runs pareados (Isaac convertido por ``isaac_rec_convert.py`` e Vials-Lite com
``--scenes`` das mesmas cenas), casando os episódios pelo índice:

1. **Diferença entre simuladores** no critério do workshop: tabela 2×2 dos pares, IC 95% da
   diferença pareada (``evalkit.stats.paired_difference``) e McNemar exato.
2. **O mesmo no critério "entrou no furo"**: sucesso do workshop com o frasco encaixado ou no furo
   (``success_seating.classify``), sem os sobre o tampo nem os pré-colocados.
3. **Ordem entre políticas** em cada simulador e a diferença pareada pi0.5 − ACT do curso dentro
   de cada um (as políticas veem as mesmas cenas: os ``*.scenes.jsonl`` do Isaac são idênticos).

As diferenças são Isaac − Vials-Lite (positivo: o Vials-Lite fica abaixo).

Uso (na raiz do repositório, com os runs do evalkit em ``outputs/evalkit``)::

    python scripts/paired_compare.py            # tabelas em Markdown
    python scripts/paired_compare.py --root outputs/evalkit
    python scripts/paired_compare.py --physics d28   # a física anterior
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from success_seating import classify

from evalkit.stats import mcnemar_exact, paired_difference

POLICIES = {"act_curso": "ACT do curso", "pi05": "pi0.5", "act_isaac": "ACT-Isaac", "act_lite": "ACT-Lite",
            "groot": "GR00T N1.6", "groot_simonly": "GR00T N1.6 só sim"}
VARIANTS = ("Eval", "DR-Eval")
IN_HOLE = ("encaixado", "no_furo")


def pairs_1984(root: Path, physics: str = "d33") -> list[tuple[str, str, str, Path, Path]]:
    """As células da comparação pareada (cenas da 2ª rodada do Isaac, semente 1984) com a física ``physics``
    (prefixo das pastas: d28, d33): as 8 das políticas do LeRobot e as do GR00T, quando os dois lados
    já rodaram."""
    cells = [
        ("1984", p, v, root / "isaac" / f"{p}_{v}", root / f"{physics}_isaac_scenes" / f"{p}_{v}")
        for p in POLICIES
        for v in VARIANTS
    ]
    return [c for c in cells if (c[3] / "episodes.jsonl").exists() and (c[4] / "episodes.jsonl").exists()]


def pairs_2026(root: Path, rep: int = 0, physics: str = "d33") -> list[tuple[str, str, str, Path, Path]]:
    """Semente 2026: Isaac 1ª rodada (a única com ``steps/``) × o run ``rep`` do pi0.5 no Vials-Lite.

    As outras políticas entram quando os dois lados já rodaram; o GR00T tem 2 runs no Vials-Lite
    (``rep`` 0 e 1)."""
    out = []
    for v in VARIANTS:
        isaac, lite = root / "isaac", root / f"{physics}_s2026"
        out.append(("2026", "act_curso", v, isaac / f"act_curso_s2026_{v}", lite / f"act_curso_{v}"))
        out.append(("2026", "pi05", v, isaac / f"pi05_s2026_{v}", lite / f"pi05_{v}_rep{rep}"))
        for p in ("act_isaac", "act_lite", "groot", "groot_simonly"):
            suffix = f"_rep{rep}" if p.startswith("groot") and rep else ""
            cell = ("2026", p, v, isaac / f"{p}_s2026_{v}", lite / f"{p}_{v}{suffix}")
            if (cell[3] / "episodes.jsonl").exists() and (cell[4] / "episodes.jsonl").exists():
                out.append(cell)
    return out


def outcomes(run: Path, in_hole: bool = False) -> dict[int, bool]:
    """Resultado por episódio: sucesso do workshop ou, com ``in_hole``, sucesso com o frasco no furo."""
    res = {}
    for line in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        ep = json.loads(line)
        ok = bool(ep["success"])
        if ok and in_hole:
            ok = classify(run / "steps" / f"ep_{ep['episode']:03d}.npz", ep)[0] in IN_HOLE
        res[ep["episode"]] = ok
    return res


def table(x: dict[int, bool], y: dict[int, bool]) -> tuple[int, int, int, int]:
    """Tabela 2×2 dos pares (os dois, só x, só y, nenhum) nos episódios comuns."""
    ks = sorted(set(x) & set(y))
    both = sum(x[k] and y[k] for k in ks)
    only_x = sum(x[k] and not y[k] for k in ks)
    only_y = sum(y[k] and not x[k] for k in ks)
    return both, only_x, only_y, len(ks) - both - only_x - only_y


def fmt_pts(x: float) -> str:
    return f"{100 * x:+.0f}".replace("-", "−")


def fmt_diff(t: tuple[int, int, int, int]) -> str:
    d = paired_difference(*t)
    return f"{fmt_pts(d.estimate)} [{fmt_pts(d.low)}; {fmt_pts(d.high)}]"


def fmt_p(p: float) -> str:
    return "p < 0,001" if p < 0.001 else "p = " + f"{p:.3f}".replace(".", ",")


def sim_vs_sim(cells, in_hole: bool) -> list[float]:
    """Imprime a tabela Isaac × Vials-Lite e devolve a largura de cada IC (pontos)."""
    print(
        "| semente | política | variante | Isaac | Vials-Lite | só Isaac / só VL | concordância "
        "| Isaac − VL (IC 95% pareado) | McNemar |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    widths = []
    for seed, p, v, isaac, lite in cells:
        a, b = outcomes(isaac, in_hole), outcomes(lite, in_hole)
        t = table(a, b)
        n = sum(t)
        d = paired_difference(*t)
        widths.append(100 * max(abs(d.low), abs(d.high)))
        print(
            f"| {seed} | {POLICIES[p]} | {v} | {t[0] + t[1]} | {t[0] + t[2]} | {t[1]} / {t[2]} "
            f"| {100 * (t[0] + t[3]) / n:.0f}% | {fmt_diff(t)} | {fmt_p(mcnemar_exact(t[1], t[2]))} |"
        )
    return widths


def kendall_tau(xs: list[float], ys: list[float]) -> float:
    """τ-b de Kendall (empates descontados), para a ordem das políticas nos dois simuladores."""
    conc = disc = tx = ty = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            dx, dy = xs[i] - xs[j], ys[i] - ys[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx * dy > 0:
                conc += 1
            else:
                disc += 1
    den = ((conc + disc + tx) * (conc + disc + ty)) ** 0.5
    return (conc - disc) / den if den else float("nan")


def ranking(cells) -> None:
    """Ordem das políticas por simulador e pi0.5 − ACT do curso pareado dentro de cada um."""
    by = {}
    for seed, p, v, isaac, lite in cells:
        by.setdefault((seed, v), {})[p] = (outcomes(isaac), outcomes(lite))
    print(
        "| semente | variante | ordem no Isaac | ordem no Vials-Lite | τ de Kendall "
        "| pi0.5 − ACT do curso, Isaac | pi0.5 − ACT do curso, Vials-Lite |"
    )
    print("|---|---|---|---|---|---|---|")
    for (seed, v), pol in by.items():
        names = list(pol)
        cols = []
        for side in (0, 1):
            rates = {p: sum(pol[p][side].values()) for p in names}
            order = sorted(names, key=lambda p: -rates[p])
            cols.append((rates, " > ".join(f"{POLICIES[p]} {rates[p]}" for p in order)))
        tau = kendall_tau([cols[0][0][p] for p in names], [cols[1][0][p] for p in names])
        diffs = []  # pi0.5 − ACT: a tabela (os dois, só pi0.5, só ACT, nenhum)
        for side in (0, 1):
            t = table(pol["pi05"][side], pol["act_curso"][side])
            diffs.append(f"{fmt_diff(t)} ({fmt_p(mcnemar_exact(t[1], t[2]))})")
        tau_s = f"{tau:+.2f}".replace(".", ",")
        print(f"| {seed} | {v} | {cols[0][1]} | {cols[1][1]} | {tau_s} | {diffs[0]} | {diffs[1]} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=Path("outputs/evalkit"))
    ap.add_argument("--rep", type=int, default=0, help="run do pi0.5 no Vials-Lite na semente 2026 (0, 1, 2)")
    ap.add_argument("--physics", default="d33", help="física do Vials-Lite, prefixo das pastas (d28, d33)")
    args = ap.parse_args()
    cells_1984 = pairs_1984(args.root, args.physics)
    cells = cells_1984 + pairs_2026(args.root, args.rep, args.physics)

    print("## 1. Critério do workshop: Isaac × Vials-Lite, pareado\n")
    w = sim_vs_sim(cells, in_hole=False)
    n = len(cells_1984)
    print(f"\nMaior |limite| do IC: semente 1984 {max(w[:n]):.1f} pontos; semente 2026 {max(w[n:]):.1f}.\n")

    print('## 2. Critério "entrou no furo": Isaac × Vials-Lite, pareado\n')
    sim_vs_sim(cells, in_hole=True)

    print("\n## 3. Ordem entre políticas em cada simulador\n")
    ranking(cells)


if __name__ == "__main__":
    main()
