"""Onde o frasco termina nos episódios de sucesso.

O critério de sucesso do workshop (``vials_lite.success``) não verifica se o frasco entrou num furo:
pede o frasco solto, com |z do frasco · z do mundo| > 0,7 (até ~45°), dentro da caixa do rack e com
a origem abaixo de 10 cm no frame do rack. Em pé sobre o tampo, com a ponta na borda de um furo, a
origem fica em ~0,09 m (``RACK_TOP`` 0,073 − ``VIAL_TIP_Z`` −0,017) e o episódio conta como sucesso.

Para cada episódio de sucesso de um run do evalkit (Vials-Lite ou Isaac convertido por
``isaac_rec_convert.py``), olha o último step do ``steps/ep_XXX.npz``, pega os frascos que satisfazem a
geometria do critério e classifica pela altura da origem do frasco no frame do rack:

* ``encaixado``: z < 0,03 (assentado no fundo do furo, ~0,016–0,022);
* ``no_furo``: 0,03 ≤ z < 0,08 (entrou no furo ou no chanfro, em geral inclinado 15–45°);
* ``sobre_tampo``: z ≥ 0,08 (em pé sobre o tampo, sem entrar);
* ``pre_colocado``: só o frasco que já começa no rack satisfaz a geometria (a peculiaridade do
  critério descrita em ``vials_lite.success``).

Os frascos pré-colocados ficam de fora quando há outro frasco válido. O último step é o da
confirmação do sucesso (o episódio termina ali), então o frasco ainda pode estar descendo.

Uso::

    python scripts/success_seating.py outputs/evalkit/<run> [<run> ...]
    python scripts/success_seating.py --episodes outputs/evalkit/<run>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vials_lite.success import quat_inverse, quat_rotate, vial_geometry_ok

SEATED_MAX_Z = 0.03
ON_TOP_MIN_Z = 0.08
CLASSES = ("encaixado", "no_furo", "sobre_tampo", "pre_colocado")


def _pre_placed(ep: dict) -> list[int]:
    pre = ep.get("vial_on_rack_at_reset")
    if pre is None:
        return []
    return [pre] if isinstance(pre, int) else list(pre)


def classify(npz: Path, ep: dict) -> tuple[str, float | None]:
    """Classe e altura (frame do rack) do frasco que sustenta o sucesso no último step."""
    f = np.load(npz)
    rack_pos, rack_quat = f["rack_pos"][-1], f["rack_quat"][-1]
    valid = []
    for v in range(f["vial_pos"].shape[1]):
        pos, quat = f["vial_pos"][-1, v], f["vial_quat"][-1, v]
        if all(vial_geometry_ok(pos, quat, rack_pos, rack_quat)):
            local = quat_rotate(quat_inverse(rack_quat), pos - rack_pos)
            valid.append((v, float(local[2])))
    pre = _pre_placed(ep)
    own = [z for v, z in valid if v not in pre]
    if not own:
        return "pre_colocado", None
    z = min(own)
    if z < SEATED_MAX_Z:
        return "encaixado", z
    return ("no_furo" if z < ON_TOP_MIN_Z else "sobre_tampo"), z


def summarize(run: Path, per_episode: bool = False) -> dict[str, int]:
    counts = dict.fromkeys(CLASSES, 0)
    for line in (run / "episodes.jsonl").read_text().splitlines():
        ep = json.loads(line)
        if not ep["success"]:
            continue
        cls, z = classify(run / "steps" / f"ep_{ep['episode']:03d}.npz", ep)
        counts[cls] += 1
        if per_episode:
            zs = f"  z = {z:.3f}" if z is not None else ""
            print(f"  ep {ep['episode']:3d}  {cls:12s}{zs}")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="runs do evalkit (com episodes.jsonl e steps/)")
    ap.add_argument("--episodes", action="store_true", help="lista a classe de cada episódio de sucesso")
    args = ap.parse_args()
    print(f"{'run':44s} {'sucessos':>8s} " + " ".join(f"{c:>12s}" for c in CLASSES))
    for run in args.runs:
        if not (run / "episodes.jsonl").exists() or not (run / "steps").is_dir():
            continue
        if args.episodes:
            print(run)
        c = summarize(run, args.episodes)
        print(f"{str(run):44s} {sum(c.values()):8d} " + " ".join(f"{c[k]:12d}" for k in CLASSES))


if __name__ == "__main__":
    main()
