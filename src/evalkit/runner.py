"""Runner do evalkit: N episódios de uma política no Vials-Lite com estado privilegiado.

Protocolo do ``lerobot_eval.py`` do workshop: a cada episódio, ``reset`` com a semente
``seed + i``; nos primeiros ``spec.TIMING.warmup_steps_eval`` (10) steps o braço recebe a ação
inicial e a política não é consultada; depois, uma ação da política por step até o sucesso
(``terminated``) ou o fim dos 450 steps (o aquecimento conta dentro deles).

Saídas em ``outputs/evalkit/<run_id>/``:
    run.json          política, variante, semente, versões e o resumo (taxa, IC de Wilson)
    episodes.jsonl    uma linha por episódio: sucesso, fase máxima, categoria de falha, ...
    steps/ep_000.npz  por step: juntas, ação, fase, posição da garra, poses e forças
    videos/ep_000.mp4 câmera externa (opcional, ``--video N`` primeiros episódios)
    report.md         ``evalkit.report``

Reclassificar um run gravado depois de revisar fases ou taxonomia: ``--rescore <pasta do run>``.

Cenas do dataset: ``--scenes outputs/scenes/scenes.jsonl`` roda um episódio por
cena reconstruída (``scripts/reconstruct_scenes.py``), com o braço partindo de ``state[0]`` do
episódio do dataset e o aquecimento segurando essa pose. ``--replay <pasta do dataset>`` troca a
política pelas ações gravadas em malha aberta (``ReplayPolicy``, sem aquecimento): um teste da
física e da reconstrução que não depende de política nenhuma. Com a pasta ``<variante>.rec`` do
gravador do Isaac (``scripts/isaac_recorder.py``) e o ``scenes.jsonl`` que o
``scripts/isaac_rec_convert.py`` tira dela, repete aqui as ações de cada episódio do Isaac, na mesma
cena (``IsaacReplayPolicy``): com as mesmas ações, as diferenças de resultado entre os dois
simuladores vêm da física, não da visão da política.

Ablações de cena: ``--scenes ... --ablate vials,joint_pos`` troca fatores da cena
do dataset pelos do reset do workshop (``ABLATIONS``), um de cada vez ou juntos, para separar o que
pesa na diferença entre a taxa nas cenas do dataset e a da ``eval``.

Arranjo do dataset: ``--layout dataset`` sorteia como a ``--variant``, mas com a
lightbox, o tapete e as faixas do rack e dos frascos no arranjo do dataset do curso em relação ao robô
(``spec.DATASET_LAYOUT_OFFSET``); a ``eval`` e a ``dr-eval`` de sempre são ``--layout workshop``.
``--init-pose dataset`` parte da pose inicial dos operadores (``spec.DATASET_INITIAL_ACTION``, que o
aquecimento segura) e ``--external-cam fixed`` tira o sorteio da câmera externa. As
três opções são independentes e não mudam os números sorteados: com a mesma ``--seed``,
os episódios de condições diferentes ficam pareados.

Ablação da ``dr-eval``: ``--variant dr-eval --dr-components dome`` liga só os extras
dados (``DR_COMPONENTS``: ``dome``, ``robot_color``, ``mat_yaw``; ``none`` desliga todos). Os sorteios
dos desligados são consumidos e descartados: com a mesma ``--seed``, cada combinação fica pareada com
a ``dr-eval`` completa (mesma cor, luz e giro sorteados) e, na cena física, com a ``eval``.

Servidor no protocolo do GR00T: ``--policy groot://host:porta`` consulta um
servidor de política (``evalkit.groot_bridge``, ou um GR00T de verdade) como o ``lerobot_eval.py`` do
workshop, executando as primeiras ``--action-horizon`` ações de cada chunk. ``--rename-map`` troca o nome
das câmeras, como o ``--rename_map`` do workshop (o GR00T do curso espera ``front`` e ``wrist``).

Semente da política: a ``--seed`` fixa a cena, mas políticas com amostragem (o flow matching do
pi0.5) também sorteiam ruído na inferência. Antes de cada episódio, o runner chama ``policy.seed(...)``
com ``policy_seed(seed + i, rep)`` (no ``LeRobotPolicy``, ``torch.manual_seed``). Isso fixa o ruído, mas
não torna o run reproduzível: o render da NVIDIA varia em ±1 nível em ~0,05% dos pixels de um ``reset``
para outro, mesmo com a mesma semente, e essa diferença basta para mudar a trajetória do pi0.5 (e,
raramente, a do ACT). ``--rep N`` (padrão 0) troca o sorteio da política com as mesmas cenas: N
repetições pareadas para medir a variação entre runs. As políticas por servidor (``groot://``) sorteiam
do lado do servidor e não são fixadas.

Render: no Linux, sem ``MUJOCO_GL`` definido, o runner usa ``egl``; antes do
primeiro episódio, confere ``GL_RENDERER`` e recusa rodar fora da NVIDIA (Mesa/Intel/llvmpipe dão
~85 s por episódio em vez de ~1,7 s, e imagens que não batem com as dos outros runs), a menos que
se passe ``--any-renderer``. A placa fica em ``run.json`` e no relatório.

Uso:
    uv run python -m evalkit.runner --policy checkpoints/act_vials_50k --variant eval \\
        --episodes 20 --seed 1000 --video 4
    uv run --with pyarrow python -m evalkit.runner --replay datasets/so101_teleop_vials_rack_left \\
        --scenes outputs/scenes/scenes.jsonl --video 4
    uv run python -m evalkit.runner --policy checkpoints/act_vials_50k \\
        --scenes outputs/scenes/scenes.jsonl --ablate vials,joint_pos
    uv run python -m evalkit.runner --policy checkpoints/act_vials_50k --variant eval \\
        --layout dataset --init-pose dataset --external-cam fixed --episodes 100
    uv run python -m evalkit.runner --policy checkpoints/act_vials_50k --variant dr-eval \
        --dr-components dome --episodes 100
    uv run python -m evalkit.runner --policy groot://localhost:5555 --action-horizon 100 --variant eval
    uv run python -m evalkit.runner --policy groot://localhost:5555 --action-horizon 16 \
        --rename-map '{"external_D455": "front", "ego": "wrist"}' --scenes Eval.scenes.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from vials_lite import spec
from vials_lite.joint_mapping import radians_to_lerobot

from .grasps import closures_from_arrays
from .progress import ProgressState, ProgressTracker
from .stats import wilson_interval
from .taxonomy import classify

INITIAL_ACTION = radians_to_lerobot(np.array(spec.INITIAL_JOINT_POS)).astype(np.float32)

# Fatores da cena do dataset que ``--ablate`` troca pelos do workshop. "vials": os frascos deitados
# são sorteados como no reset (``spec.RESET.vial_pose``; os que começam no rack ficam); "joint_pos":
# o braço parte da pose inicial do workshop (``spec.INITIAL_JOINT_POS``); os demais tiram as chaves da
# cena, e o env usa o que sorteou para a semente do episódio (a distribuição da ``--variant``).
ABLATIONS: dict[str, tuple[str, ...]] = {
    "vials": ("vial_pos", "vial_quat"),
    "joint_pos": ("joint_pos",),
    "mat": ("mat_yaw", "mat_offset", "mat_half_size"),
    "cameras": ("external_cam_pos", "external_cam_quat", "external_cam_fovy", "ego_fovy"),
    "lighting": ("lighting", "robot_color"),
}


def ablate_scenes(scenes: list[dict], factors: list[str], seed: int) -> list[dict]:
    """Cópias das linhas de ``scenes.jsonl`` com os ``factors`` (chaves de ``ABLATIONS``) trocados
    pelos do workshop. ``seed`` fixa o sorteio dos frascos (em ordem, cena a cena)."""
    unknown = set(factors) - set(ABLATIONS)
    if unknown:
        raise ValueError(f"fatores desconhecidos: {sorted(unknown)}; use {sorted(ABLATIONS)}")
    rng = np.random.default_rng(seed)
    out = []
    for row in scenes:
        scene = json.loads(json.dumps(row["scene"]))  # cópia profunda
        for factor in factors:
            if factor == "vials":
                if "vial_pos" not in scene:
                    continue
                from vials_lite.env import sample_lying_vial

                in_rack = set(scene.get("vials_in_rack", []))
                for i in range(len(spec.VIALS)):
                    if i not in in_rack:
                        pos, quat = sample_lying_vial(rng, i)
                        scene["vial_pos"][i], scene["vial_quat"][i] = pos.tolist(), quat.tolist()
            elif factor == "joint_pos":
                scene["joint_pos"] = INITIAL_ACTION.tolist()  # o aquecimento segura essa pose
            else:
                for key in ABLATIONS[factor]:
                    scene.pop(key, None)
        out.append({**row, "scene": scene})
    return out


def probe(info: dict[str, Any], success: bool) -> ProgressState:
    """Probe do Vials-Lite: ``info["privileged"]`` → ``ProgressState``."""
    p = info["privileged"]
    return ProgressState(
        gripper_pos=p["gripper_pos"],
        vial_pos=p["vial_pos"],
        vial_quat=p["vial_quat"],
        rack_pos=p["rack_pos"],
        rack_quat=p["rack_quat"],
        jaw_contact_force=p["jaw_contact_force"],
        success=success,
    )


class ReplayPolicy:
    """Ações gravadas de um dataset LeRobot em malha aberta, por episódio do dataset.

    O gravador do workshop chama ``step(action[t])`` e depois lê ``observation.state[t]``; a cena
    reconstruída parte de ``state[0]``, então o replay começa em ``action[1]``.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.name = f"replay:{self.data_dir.name}"
        self._actions: np.ndarray | None = None
        self._k = 0

    def set_episode(self, episode: int) -> None:
        import pyarrow.parquet as pq

        table = pq.read_table(self.data_dir / "data" / "chunk-000" / f"file-{episode:03d}.parquet",
                              columns=["action"])
        self._actions = np.stack(table.column("action").to_numpy(zero_copy_only=False)).astype(np.float32)
        self._k = 0

    def reset(self) -> None:
        self._k = 0

    def act(self, observation, instruction) -> np.ndarray:
        self._k = min(self._k + 1, len(self._actions) - 1)  # depois do fim, segura a última
        return self._actions[self._k]


def isaac_joint_order(names: list[str]) -> list[int]:
    """Índices das juntas de um registro do Isaac na ordem do LeRobot.

    O Isaac grava os nomes do USD (``spec.JOINTS_USD``: ``Rotation``, ``Pitch``...); os do LeRobot
    (``spec.JOINTS_LEROBOT``, iguais aos do MJCF) também são aceitos, como nos testes.
    """
    usd_to_lerobot = dict(zip(spec.JOINTS_USD, spec.JOINTS_LEROBOT, strict=True))
    canon = [usd_to_lerobot.get(n, n) for n in names]
    return [canon.index(j) for j in spec.JOINTS_LEROBOT]


class IsaacReplayPolicy:
    """Ações gravadas no Isaac pelo ``scripts/isaac_recorder.py``, em malha aberta.

    ``data_dir`` é a pasta ``<variante>.rec``; a cena vem do ``scenes.jsonl`` do
    ``scripts/isaac_rec_convert.py``, cujo ``episode`` é o do registro. A ação *k* do registro é a que
    o cliente mandou no step *k* depois do reset, aquecimento incluído; ela vem em radianos, na ordem
    das juntas do Isaac (``meta.json``), e sai no espaço do LeRobot.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.name = f"replay-isaac:{self.data_dir.parent.name}/{self.data_dir.name}"
        names = json.loads((self.data_dir / "meta.json").read_text(encoding="utf-8"))["joint_names"]
        self._order = isaac_joint_order(names)
        self._actions: np.ndarray | None = None
        self._k = 0

    @staticmethod
    def is_recording(path: Path) -> bool:
        return (Path(path) / "meta.json").is_file() and (Path(path) / "episodes.jsonl").is_file()

    def set_episode(self, episode: int) -> None:
        rad = np.load(self.data_dir / f"ep_{episode:04d}.npz")["action"][:, self._order]
        self._actions = radians_to_lerobot(rad).astype(np.float32)
        self._k = 0

    def reset(self) -> None:
        self._k = 0

    def act(self, observation, instruction) -> np.ndarray:
        a = self._actions[min(self._k, len(self._actions) - 1)]  # depois do fim, segura a última
        self._k += 1
        return a


REP_STRIDE = 1_000_000


def policy_seed(seed: int, rep: int = 0) -> int:
    """Semente da política no episódio de semente ``seed``, na repetição ``rep`` (a 0 usa a própria)."""
    return seed + REP_STRIDE * rep


def run_episode(env, policy, seed: int, warmup_steps: int, max_steps: int | None = None, frames=None,
                scene: dict | None = None, rep: int = 0) -> dict:
    """Um episódio; devolve o resumo e os arrays por step. ``scene``: linha de ``scenes.jsonl``."""
    options = None
    hold = getattr(env, "initial_action", INITIAL_ACTION)
    if scene is not None:
        options = {"scene": scene["scene"]}
        hold = np.asarray(scene["scene"]["joint_pos"], np.float32)
        if hasattr(policy, "set_episode"):
            policy.set_episode(scene["episode"])
    obs, info = env.reset(seed=seed, options=options)
    if hasattr(policy, "seed"):
        policy.seed(policy_seed(seed, rep))
    policy.reset()
    ignore = info["privileged"]["vial_on_rack_at_reset"]
    tracker = ProgressTracker(n_vials=len(spec.VIALS), ignore_vial=ignore)
    # o tracker ao vivo dá a fase por step; o resumo final sai de ``summarize`` (o mesmo do ``rescore``)
    instruction = env.task_description
    steps: dict[str, list] = {k: [] for k in ("agent_pos", "action", "phase", "gripper_pos", "vial_pos",
                                               "vial_quat", "rack_pos", "rack_quat", "jaw_contact_force")}
    events: list[tuple[int, str]] = []
    n = max_steps or spec.TIMING.episode_steps
    terminated = truncated = False
    t0 = time.perf_counter()
    for k in range(n):
        action = hold if k < warmup_steps else np.asarray(policy.act(obs, instruction), np.float32)
        obs, _, terminated, truncated, info = env.step(action)
        tracker.update(probe(info, terminated))
        p = info["privileged"]
        steps["agent_pos"].append(obs["agent_pos"])
        steps["action"].append(action)
        steps["phase"].append(tracker.phases[-1])
        for key in ("gripper_pos", "vial_pos", "vial_quat", "rack_pos", "rack_quat", "jaw_contact_force"):
            steps[key].append(p[key])
        events += [(p["step"], e) for e in p["events"]]
        if frames is not None:
            frames.append(obs["pixels"]["external_D455"] if "pixels" in obs else env.render())
        if terminated or truncated:
            break
    arrays = {k: np.asarray(v) for k, v in steps.items()}
    return {
        "seed": seed,
        "policy_seed": policy_seed(seed, rep) if hasattr(policy, "seed") else None,
        "success": bool(terminated),
        "steps": k + 1,
        **summarize(arrays, bool(terminated), ignore),
        "vial_on_rack_at_reset": ignore,
        "robot_color": info["privileged"]["robot_color"],
        "mat_yaw": info["privileged"]["mat_yaw"],
        "lighting": info["privileged"]["lighting"],
        "events": events,
        "wall_s": round(time.perf_counter() - t0, 2),
        "_arrays": arrays,
    }


def summarize(arrays: dict[str, np.ndarray], success: bool, ignore_vial) -> dict:
    """Fase máxima, categoria de falha e fechamentos da garra (``grasps``) a partir dos arrays por
    step (também usado por ``rescore``)."""
    tracker = ProgressTracker(n_vials=arrays["vial_pos"].shape[1], ignore_vial=ignore_vial)
    n = len(arrays["phase"])
    for k in range(n):
        fields = ("gripper_pos", "vial_pos", "vial_quat", "rack_pos", "rack_quat", "jaw_contact_force")
        tracker.update(ProgressState(**{f: arrays[f][k] for f in fields}, success=success and k == n - 1))
    s = tracker.summary(success)
    return {
        "max_phase": s.max_phase.name,
        "failure": classify(s).value,
        "n_grasps": s.n_grasps,
        "n_releases_outside_rack": s.n_releases_outside_rack,
        "released_inside_rack_tilted": s.released_inside_rack_tilted,
        "released_on_rack_top": s.released_on_rack_top,
        "min_vial_to_slot_dist_m": None if np.isinf(s.min_vial_to_slot_dist_m)
        else round(s.min_vial_to_slot_dist_m, 4),
        "closures": closures_from_arrays(arrays, ignore_vial),
    }


def rescore(run_dir: Path) -> None:
    """Reclassifica um run já gravado (fases e taxonomia revistas) sem simular de novo."""
    from .report import load_run, write_report

    _, episodes = load_run(run_dir)
    with (run_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for ep in episodes:
            arrays = dict(np.load(run_dir / "steps" / f"ep_{ep['episode']:03d}.npz"))
            ep.update(summarize(arrays, ep["success"], ep["vial_on_rack_at_reset"]))
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")
    write_report(run_dir)


def write_video(path: Path, frames: list[np.ndarray], fps: int = 60) -> None:
    import av

    with av.open(str(path), "w") as c:
        stream = c.add_stream("libx264", rate=fps)
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        for f in frames:
            c.mux(stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")))
        c.mux(stream.encode())


def _versions() -> dict[str, str]:
    import importlib.metadata as md
    import subprocess

    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("mujoco", "lerobot", "torch", "transformers"):  # transformers: VLAs (pi0.5)
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    try:
        out["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                       text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return out


def run(policy, env, n_episodes: int, seed: int, out_dir: Path, warmup_steps: int | None = None,
        n_video: int = 0, max_steps: int | None = None, meta: dict | None = None,
        scenes: list[dict] | None = None, rep: int = 0) -> dict:
    """``scenes``: linhas de ``scenes.jsonl``, uma por episódio (então ``n_episodes`` ≤ len).
    ``rep``: repetição do sorteio da política (``policy_seed``), com as mesmas cenas."""
    warmup = spec.TIMING.warmup_steps_eval if warmup_steps is None else warmup_steps
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "steps").mkdir(exist_ok=True)
    if n_video:
        (out_dir / "videos").mkdir(exist_ok=True)
    episodes = []
    with (out_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for i in range(n_episodes):
            frames: list[np.ndarray] | None = [] if i < n_video else None
            scene = scenes[i] if scenes is not None else None
            ep = run_episode(env, policy, seed + i, warmup, max_steps, frames, scene, rep)
            if scene is not None:
                ep["dataset_episode"] = scene["episode"]
            np.savez_compressed(out_dir / "steps" / f"ep_{i:03d}.npz", **ep.pop("_arrays"))
            if frames:
                write_video(out_dir / "videos" / f"ep_{i:03d}.mp4", frames)
            ep["episode"] = i
            episodes.append(ep)
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")
            f.flush()
            ok = sum(e["success"] for e in episodes)
            print(f"ep {i:3d} seed {ep['seed']}: {'sucesso' if ep['success'] else ep['failure']}"
                  f" (fase {ep['max_phase']}, {ep['steps']} steps, {ep['wall_s']} s) — {ok}/{i + 1}")
    ok = sum(e["success"] for e in episodes)
    ci = wilson_interval(ok, n_episodes)
    run_info = {
        **(meta or {}),
        "n_episodes": n_episodes,
        "seed": seed,
        "rep": rep,
        "warmup_steps": warmup,
        "successes": ok,
        "success_rate": ok / n_episodes,
        "wilson_95": [ci.low, ci.high],
        "finished": datetime.now().isoformat(timespec="seconds"),
        "versions": _versions(),
    }
    (out_dir / "run.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")
    from .report import write_report

    write_report(out_dir)
    return run_info


def is_gpu_renderer(gl_renderer: str) -> bool:
    """Render na NVIDIA (o dos runs registrados)? Intel, Mesa/llvmpipe e "desconhecido" não contam."""
    return "nvidia" in gl_renderer.lower()


def main() -> None:
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")  # antes de importar o mujoco (via vials_lite.env)
    from vials_lite.env import DR_COMPONENTS, EXTERNAL_CAMS, INIT_POSES, LAYOUTS, VARIANTS, VialsLiteEnv

    from .groot_bridge import GrootRemotePolicy, parse_groot_url
    from .lerobot_policy import LeRobotPolicy

    ap = argparse.ArgumentParser(description="Avalia uma política LeRobot no Vials-Lite (evalkit).")
    ap.add_argument("--rescore", type=Path, default=None, help="só reclassifica um run já gravado")
    ap.add_argument("--policy", help="pasta ou repo do Hub de uma política LeRobot, ou groot://host:porta")
    ap.add_argument("--action-horizon", type=int, default=None,
                    help="com groot://: ações executadas por chunk (16 no protocolo do curso para o GR00T; "
                         "100 para o ACT e o pi0.5 servidos pela ponte, o chunk deles)")
    ap.add_argument("--rename-map", default=None,
                    help="com groot://: JSON com os nomes das câmeras para o servidor, como o --rename_map "
                         "do workshop")
    ap.add_argument("--replay", type=Path, default=None,
                    help="dataset LeRobot ou registro do Isaac (<variante>.rec): ações gravadas em "
                         "malha aberta (exige --scenes)")
    ap.add_argument("--scenes", type=Path, default=None,
                    help="scenes.jsonl de scripts/reconstruct_scenes.py: um episódio por cena")
    ap.add_argument("--ablate", default="",
                    help=f"com --scenes: fatores da cena trocados pelos do workshop ({','.join(ABLATIONS)})")
    ap.add_argument("--variant", default="eval", choices=VARIANTS)
    ap.add_argument("--dr-components", default=None,
                    help=f"só com --variant dr-eval: extras ligados, separados por vírgula "
                         f"({','.join(DR_COMPONENTS)}), ou none; padrão: todos")
    ap.add_argument("--layout", default="workshop", choices=LAYOUTS,
                    help="arranjo da cena em relação ao robô: o do workshop ou o do dataset")
    ap.add_argument("--init-pose", default="workshop", choices=INIT_POSES,
                    help="pose inicial do braço: a do workshop ou a dos operadores do dataset")
    ap.add_argument("--external-cam", default="random", choices=EXTERNAL_CAMS,
                    help="câmera externa sorteada em volta da calibrada (workshop) ou fixa nela")
    ap.add_argument("--instruction", default=None,
                    help="texto da tarefa (padrão: o do curso, spec.LANGUAGE_INSTRUCTION; "
                         "'dataset' = spec.DATASET_TASK)")
    ap.add_argument("--actuators", default="isaac", choices=("isaac", "menagerie"))
    ap.add_argument("--episodes", type=int, default=None, help="padrão: 20, ou todas as --scenes")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--rep", type=int, default=0,
                    help="repetição do sorteio da política (pi0.5), com as mesmas cenas da --seed")
    ap.add_argument("--warmup", type=int, default=None, help="padrão: 10, como o lerobot_eval.py")
    ap.add_argument("--video", type=int, default=0, help="grava a câmera externa dos N primeiros episódios")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--any-renderer", action="store_true",
                    help="roda mesmo com o render fora da NVIDIA (lento e não comparável aos outros runs)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.rescore:
        rescore(args.rescore)
        print(f"reclassificado: {args.rescore / 'report.md'}")
        return
    if bool(args.policy) == bool(args.replay):
        ap.error("use --policy ou --replay (ou --rescore)")
    if args.replay and not args.scenes:
        ap.error("--replay exige --scenes (a cena do episódio gravado)")
    ablate = [f for f in args.ablate.split(",") if f]
    if ablate and not args.scenes:
        ap.error("--ablate exige --scenes")
    if args.scenes and args.layout != "workshop":
        ap.error("--layout não se aplica a --scenes (as cenas já trazem o arranjo do dataset)")
    if args.scenes and (args.init_pose != "workshop" or args.external_cam != "random"):
        ap.error("--init-pose e --external-cam não se aplicam a --scenes (as cenas trazem pose e câmera)")
    if set(ablate) - set(ABLATIONS):
        ap.error(f"--ablate: use {','.join(ABLATIONS)}")
    dr_components = None
    if args.dr_components is not None:
        if args.variant != "dr-eval" or args.scenes:
            ap.error("--dr-components exige --variant dr-eval e não se aplica a --scenes")
        dr_components = [c for c in args.dr_components.split(",") if c and c != "none"]
        if set(dr_components) - set(DR_COMPONENTS):
            ap.error(f"--dr-components: use {','.join(DR_COMPONENTS)} ou none")
    groot = parse_groot_url(args.policy) if args.policy else None
    if args.rep < 0:
        ap.error("--rep: use 0 ou mais")
    if bool(groot) != (args.action_horizon is not None):
        ap.error("--action-horizon vai junto com --policy groot://host:porta (e só com ele)")
    if args.rename_map and not groot:
        ap.error("--rename-map só vale com --policy groot://host:porta")

    scenes = None
    if args.scenes:
        with args.scenes.open(encoding="utf-8") as f:
            scenes = [json.loads(line) for line in f if line.strip()]
        if ablate:
            scenes = ablate_scenes(scenes, ablate, args.seed)
    n_episodes = args.episodes or (len(scenes) if scenes else 20)
    if scenes and n_episodes > len(scenes):
        ap.error(f"--episodes {n_episodes} > {len(scenes)} cenas em {args.scenes} (uma cena por episódio)")
    if args.replay:
        replay_cls = IsaacReplayPolicy if IsaacReplayPolicy.is_recording(args.replay) else ReplayPolicy
        policy, warmup, source = replay_cls(args.replay), 0, args.replay
    elif groot:
        rename_map = json.loads(args.rename_map) if args.rename_map else None
        policy = GrootRemotePolicy(*groot, action_horizon=args.action_horizon, rename_map=rename_map)
        warmup, source = args.warmup, args.policy
    else:
        policy, warmup, source = LeRobotPolicy(args.policy, device=args.device), args.warmup, args.policy
    instruction = spec.DATASET_TASK if args.instruction == "dataset" else args.instruction
    env = VialsLiteEnv(variant=args.variant, actuators=args.actuators, task=instruction, layout=args.layout,
                       init_pose=args.init_pose, external_cam=args.external_cam,
                       dr_components=dr_components)
    gl = env.gl_renderer()
    print(f"render: {gl} (MUJOCO_GL={os.environ.get('MUJOCO_GL', 'não definido')})")
    if not is_gpu_renderer(gl) and not args.any_renderer:
        env.close()
        ap.error(f"o render não está na NVIDIA ({gl}). No Ubuntu, confira MUJOCO_GL=egl e o driver "
                 "(docs/instalacao.md); para rodar assim mesmo, --any-renderer")
    tag = "replay" if args.replay else f"groot-{groot[1]}" if groot else Path(args.policy).name
    condition = ("scenes" + "".join(f"-{f}" for f in ablate)) if scenes else args.variant
    if not scenes:
        if dr_components is not None:
            condition += "-only-" + ("+".join(env.dr_components) or "none")
        condition += f"-layout-{args.layout}" if args.layout != "workshop" else ""
        condition += f"-pose-{args.init_pose}" if args.init_pose != "workshop" else ""
        condition += f"-cam-{args.external_cam}" if args.external_cam != "random" else ""
    condition += f"-rep{args.rep}" if args.rep else ""
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{tag}_{condition}"
    out = args.out or Path("outputs") / "evalkit" / run_id
    meta = {
        "run_id": run_id, "policy": str(source), "policy_name": policy.name,
        "variant": "scenes" if scenes else args.variant, "scenes": str(args.scenes) if scenes else None,
        "ablate": ablate, "dr_components": None if scenes else list(env.dr_components),
        "layout": None if scenes else args.layout,
        "init_pose": None if scenes else args.init_pose,
        "external_cam": None if scenes else args.external_cam,
        "instruction": env.task_description,
        "actuators": args.actuators, "gl_renderer": gl, "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "env": "vials_lite", "args": {k: str(v) for k, v in vars(args).items()},
    }
    try:
        info = run(policy, env, n_episodes, args.seed, out, warmup, args.video, meta=meta, scenes=scenes,
                   rep=args.rep)
    finally:
        env.close()
    print(f"{info['successes']}/{info['n_episodes']} = {info['success_rate']:.0%}"
          f" (IC 95% {info['wilson_95'][0]:.0%}–{info['wilson_95'][1]:.0%}); saída em {out}")


if __name__ == "__main__":
    main()
