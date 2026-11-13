"""Replay dinâmico do dataset do curso no ``VialsLiteEnv``.

Diferente do replay cinemático (``replay_dataset.py``), aqui as **ações** do dataset passam pelos
atuadores simulados, e o estado resultante é comparado com o ``observation.state`` gravado.

Alinhamento temporal: o gravador do workshop (``scripts/lerobot_agent.py``) chama
``env.step(action[t])`` e só depois lê ``joint_pos_obs``, que vira ``observation.state[t]``. Logo
o estado simulado depois do step com ``action[t]`` se compara com ``state[t]``. O episódio começa
com as juntas em ``state[0]``, velocidade zero e o step 0 não é contado.

O dataset foi gravado no Isaac (teleoperação no simulador), então ``state`` é a resposta dos
atuadores do workshop. Um preset que a reproduz deve ter erro pequeno. A diferença
``action − state`` do próprio dataset mede o erro de rastreamento do Isaac e é impressa para
comparação.

Frascos e rack são tirados do caminho (o dataset não grava as poses deles). A garra é excluída
enquanto segura o frasco (``replay_dataset.held_interval``): no Isaac o frasco a bloqueia em ~18,
aqui ela fecha no vazio.

Uso (uma linha; PowerShell e bash):
    uv run --with pyarrow python scripts/replay_dynamic.py <pasta com file-*.parquet>
        [--actuators menagerie isaac] [--episodes N]

Os parquets: ver a docstring de ``replay_dataset.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from replay_dataset import held_interval  # noqa: E402

from vials_lite import spec  # noqa: E402
from vials_lite.joint_mapping import lerobot_to_radians, usd_radians_to_mjcf  # noqa: E402

ACCEPT_DEG = 2.0  # erro por junta abaixo do qual o frame conta como reproduzido


def tracking_summary(err_deg: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Resumo por junta de |erro| (graus), ignorando as entradas com ``mask`` falso.

    err_deg, mask: (frames, juntas). Devolve média, p95, máximo e fração de frames < ``ACCEPT_DEG``.
    """
    out: dict[str, list[float]] = {"mean": [], "p95": [], "max": [], "frac_ok": []}
    for e, m in zip(np.abs(err_deg).T, mask.T, strict=True):
        e = e[m]
        out["mean"].append(e.mean())
        out["p95"].append(np.percentile(e, 95))
        out["max"].append(e.max())
        out["frac_ok"].append((e < ACCEPT_DEG).mean())
    return {k: np.array(v) for k, v in out.items()}


def _clear_objects(env) -> None:
    """Põe frascos e rack no chão, longe do braço."""
    d, ids = env.data, env.ids
    for i, a in enumerate([*ids.vial_qpos, ids.rack_qpos]):
        d.qpos[a : a + 3] = [1.0 + 0.3 * i, 1.0, 0.1]
        d.qpos[a + 3 : a + 7] = [1, 0, 0, 0]


def replay_episode(env, state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Estado simulado (radianos do USD) depois de cada ``action[t]``, t ≥ 1. Shape (T − 1, 6)."""
    mj = env._mj
    env.reset(seed=0)
    _clear_objects(env)
    q0 = usd_radians_to_mjcf(lerobot_to_radians(state[0]))
    env.data.qpos[env.ids.arm_qpos] = q0
    env.data.qvel[:] = 0.0
    env.data.ctrl[env.ids.actuators] = np.clip(
        usd_radians_to_mjcf(lerobot_to_radians(action[0])), env._ctrl_lo, env._ctrl_hi
    )
    mj.mj_forward(env.model, env.data)
    out = np.empty((len(action) - 1, 6))
    for t in range(1, len(action)):
        env.step(action[t])
        out[t - 1] = lerobot_to_radians(env.joint_positions_lerobot())
    return out


def _print_summary(label: str, s: dict[str, np.ndarray]) -> None:
    print(f"{label}")
    print(f"  {'junta':14s} {'média':>7s} {'p95':>7s} {'máx':>7s} {'<2°':>6s}")
    for k, name in enumerate(spec.JOINTS_LEROBOT):
        print(
            f"  {name:14s} {s['mean'][k]:7.2f} {s['p95'][k]:7.2f} {s['max'][k]:7.2f}"
            f" {100 * s['frac_ok'][k]:5.1f}%"
        )


def main() -> None:
    import pyarrow.parquet as pq

    from vials_lite.env import VialsLiteEnv
    from vials_lite.scene import ACTUATOR_PRESETS

    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path, help="pasta com os file-*.parquet do dataset")
    ap.add_argument("--menagerie", type=Path, default=None)
    ap.add_argument("--actuators", nargs="+", default=list(ACTUATOR_PRESETS), choices=ACTUATOR_PRESETS)
    ap.add_argument("--episodes", type=int, default=None, help="só os N primeiros episódios")
    args = ap.parse_args()

    episodes = []
    for path in sorted(args.data.glob("file-*.parquet"))[: args.episodes]:
        table = pq.read_table(path, columns=["observation.state", "action"])
        state = np.array(table.column("observation.state").to_pylist(), dtype=np.float64)
        action = np.array(table.column("action").to_pylist(), dtype=np.float64)
        mask = np.ones((len(state) - 1, 6), dtype=bool)
        interval = held_interval(state[:, 5], action[:, 5])
        if interval is not None:
            mask[max(interval[0] - 1, 0) : interval[1], 5] = False  # índices deslocados de 1
        episodes.append((state, action, mask))
    n_frames = sum(len(m) for _, _, m in episodes)
    print(f"{len(episodes)} episódios, {n_frames} frames comparados (garra fora enquanto segura)\n")

    data_state = np.concatenate([lerobot_to_radians(s[1:]) for s, _, _ in episodes])
    data_action = np.concatenate([lerobot_to_radians(a[1:]) for _, a, _ in episodes])
    mask = np.concatenate([m for _, _, m in episodes])
    _print_summary(
        "Isaac (dataset): |action − state|, graus",
        tracking_summary(np.degrees(data_action - data_state), mask),
    )

    for preset in args.actuators:
        env = VialsLiteEnv(render_images=False, menagerie_dir=args.menagerie, actuators=preset)
        sim = np.concatenate([replay_episode(env, s, a) for s, a, _ in episodes])
        env.close()
        err = np.degrees(sim - data_state)
        print()
        _print_summary(f"{preset}: |sim − state do dataset|, graus", tracking_summary(err, mask))
        _print_summary(
            f"{preset}: |action − sim| (rastreamento simulado), graus",
            tracking_summary(np.degrees(data_action - sim), mask),
        )
        bias = np.array([err[mask[:, k], k].mean() for k in range(6)])
        print(f"  viés médio sim − state (graus): {np.array2string(bias, precision=2)}")


if __name__ == "__main__":
    main()
