"""Relatório Markdown de um run do evalkit: taxa com IC, funil por fase,
categorias de falha, fechamentos da garra (``grasps``) e tabela por episódio. NumPy puro; lê só
``run.json`` e ``episodes.jsonl``.

Uso: ``python -m evalkit.report outputs/evalkit/<run_id>`` (o runner já chama no fim).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from .grasps import CLOSE_COMMAND, LIFT_MIN_RISE_M, LIFT_WINDOW_STEPS, NEAR_AXIS_M, closure_stats
from .progress import Phase
from .stats import wilson_interval
from .taxonomy import Failure


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    with (run_dir / "episodes.jsonl").open(encoding="utf-8") as f:
        episodes = [json.loads(line) for line in f if line.strip()]
    return info, episodes


def funnel(episodes: list[dict]) -> list[tuple[Phase, int]]:
    """Quantos episódios atingiram cada fase (a fase máxima implica as anteriores)."""
    reached = [Phase[e["max_phase"]] for e in episodes]
    return [(ph, sum(r >= ph for r in reached)) for ph in Phase if ph > Phase.START]


def render_report(info: dict, episodes: list[dict]) -> str:
    n = len(episodes)
    ok = sum(e["success"] for e in episodes)
    lines = [
        f"# evalkit: {info.get('policy_name', info.get('policy', '?'))} em {info.get('env', '?')}"
        f":{info.get('variant', '?')}",
        "",
        f"- Sucesso: **{ok}/{n}**, {wilson_interval(ok, n)} (IC 95% de Wilson)",
        f"- Sementes {info.get('seed')}..{info.get('seed', 0) + n - 1}; aquecimento de"
        f" {info.get('warmup_steps')} steps; atuadores `{info.get('actuators', '?')}`",
        f"- Versões: {', '.join(f'{k} {v}' for k, v in info.get('versions', {}).items())}",
        *([f"- Render: {info['gl_renderer']} (`MUJOCO_GL={info.get('mujoco_gl')}`)"]
          if info.get("gl_renderer") else []),
        *([f"- Texto da tarefa: \"{info['instruction']}\""] if info.get("instruction") else []),
        *([f"- Arranjo da cena: `{info['layout']}`"]
          if info.get("layout") not in (None, "workshop") else []),
        *([f"- Pose inicial: `{info['init_pose']}`"]
          if info.get("init_pose") not in (None, "workshop") else []),
        *([f"- Câmera externa: `{info['external_cam']}`"]
          if info.get("external_cam") not in (None, "random") else []),
        *([f"- Ablação: cenas de `{info.get('scenes')}` com {', '.join(info['ablate'])} do workshop"]
          if info.get("ablate") else []),
        "",
        "## Funil (episódios que chegaram a cada fase)",
        "",
        "| fase | episódios | % |",
        "|---|---|---|",
    ]
    lines += [f"| {ph.name} | {c} | {100 * c / n:.0f}% |" for ph, c in funnel(episodes)]
    lines += ["", "## Falhas", "", "| categoria | episódios |", "|---|---|"]
    counts = Counter(e["failure"] for e in episodes)
    lines += [f"| {f.value} | {counts[f.value]} |" for f in Failure if counts[f.value]]
    lines += closures_section(episodes)
    lines += [
        "",
        "## Episódios",
        "",
        "| ep | semente | resultado | fase máx. | steps | agarres | fechamentos (levantaram) | soltou fora"
        " | frasco no rack no reset | cor | luz |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in episodes:
        light = ", ".join(f"{k} {v:.2f}" for k, v in e.get("lighting", {}).items())
        ds = e.get("dataset_episode")
        seed = e["seed"] if ds is None else f"{e['seed']} (dataset {ds})"
        lines.append(
            f"| {e['episode']} | {seed}"
            f" | {'sucesso' if e['success'] else e['failure']}"
            f" | {e['max_phase']} | {e['steps']} | {e['n_grasps']} | {_closures_cell(e)}"
            f" | {e['n_releases_outside_rack']}"
            f" | {e['vial_on_rack_at_reset'] if e['vial_on_rack_at_reset'] is not None else '–'}"
            f" | {e.get('robot_color', '')} | {light} |"
        )
    return "\n".join(lines) + "\n"


def _mm(x: float | None) -> str:
    return "–" if x is None else f"{x:.0f} mm"


def _slope(x: float | None) -> str:
    return "–" if x is None else f"{x:.2f}"


def _closures_cell(e: dict) -> str:
    if "closures" not in e:
        return "–"
    return f"{len(e['closures'])} ({sum(c['lifted'] for c in e['closures'])})"


def closures_section(episodes: list[dict]) -> list[str]:
    """Seção "Fechamentos da garra" (vazia em runs gravados antes da análise: use ``--rescore``)."""
    st = closure_stats(episodes, Failure.NEVER_LIFTED.value)
    if st is None:
        return []
    lines = [
        "",
        "## Fechamentos da garra",
        "",
        f"Fechamento: comando da garra cruzando {CLOSE_COMMAND:g} para baixo. Levantou: o frasco mais"
        f" perto do site sobe > {100 * LIFT_MIN_RISE_M:.0f} cm em {LIFT_WINDOW_STEPS} steps."
        " Distância: do site da garra ao eixo desse frasco.",
        "",
        f"- {st.n_closures} fechamentos em {st.n_episodes} episódios (mediana de"
        f" {st.per_episode_median:g} por episódio); 1º fechamento levantou o frasco em"
        f" **{st.first_lifted}/{st.episodes_with_closure}**",
        "",
        "| fechamentos | n | distância site–eixo (mediana) |",
        "|---|---|---|",
        f"| levantaram | {st.lifted} | {_mm(st.lifted_median_dist_mm)} |",
        f"| não levantaram | {st.n_closures - st.lifted} | {_mm(st.not_lifted_median_dist_mm)} |",
        "",
        f"- Quanto a garra segue o frasco (inclinação da posição do site no 1º fechamento contra a do"
        f" frasco; 1 = segue, 0 = vai sempre ao mesmo lugar): x **{_slope(st.slope_x)}**,"
        f" y **{_slope(st.slope_y)}**",
    ]
    n_nl = st.never_lifted_closed_beside + st.never_lifted_closed_on_vial
    if n_nl:
        lines.append(
            f"- \"{Failure.NEVER_LIFTED.value}\" ({n_nl}): fechou só ao lado do frasco (nenhum fechamento"
            f" a ≤ {1000 * NEAR_AXIS_M:.0f} mm do eixo) em {st.never_lifted_closed_beside}; fechou sobre ele"
            f" e não levantou em {st.never_lifted_closed_on_vial}"
        )
    return lines


def write_report(run_dir: Path) -> Path:
    info, episodes = load_run(run_dir)
    path = run_dir / "report.md"
    path.write_text(render_report(info, episodes), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(write_report(Path(sys.argv[1])))
