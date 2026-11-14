# Arquitetura

## Visão em camadas

```
                 ┌─────────────────────────────────────────────────────────────┐
  usuários  ───▶ │ python -m evalkit.runner        avaliar, repetir, reclassificar │
                 │ python -m evalkit.report        relatório de um run         │
                 │ python -m evalkit.groot_bridge  política para o Isaac       │
                 │ lerobot-eval --env.type=vials_lite   (plugin / EnvHub)      │
                 │ scripts/   replay, reconstrução das cenas, Isaac, calibração │
                 └───────────────┬─────────────────────────────────────────────┘
                                 │
        ┌────────────────────────┴───────────────────────┐
        ▼                                                ▼
┌──────────────────────────┐               ┌───────────────────────────────┐
│ vials_lite (ambiente)    │               │ evalkit (avaliação)           │
│  env.py   VialsLiteEnv   │◀── info ──────│  runner  (loop, probe)        │
│  scene.py (MJCF, luz)    │  ["privileged"]│  lerobot_policy (ACT, pi0.5)  │
│  scripted.py (demo)      │               │  groot_bridge (ZMQ)           │
│  lerobot_env/ (plugin)   │               │  report  (report.md)          │
├──────────────────────────┤               ├───────────────────────────────┤
│ núcleo NumPy puro        │               │ núcleo NumPy puro             │
│  spec, joint_mapping,    │               │  stats, progress, taxonomy,   │
│  success                 │               │  grasps, policy (protocolos)  │
└──────────────────────────┘               └───────────────────────────────┘
        ▲ fonte única dos números da cena: spec.py (workshop e calibrações)
```

`envhub/env.py` é o arquivo publicado no EnvHub do LeRobot e só reexporta `vials_lite.lerobot_env.make_env`.

## Módulos

| Módulo | Depende de | Responsabilidade |
|---|---|---|
| `vials_lite.spec` | NumPy | todos os números da cena: tempo, poses, faixas de sorteio, câmeras, luz, limiares do sucesso |
| `vials_lite.joint_mapping` | NumPy | LeRobot (−100..100) ↔ radianos do USD ↔ radianos do MJCF |
| `vials_lite.success` | NumPy | o critério de sucesso do workshop (`SuccessTracker`), independente de simulador |
| `vials_lite.scene` | MuJoCo | monta o MJCF (Menagerie + objetos + lightbox), luz, cores, câmeras |
| `vials_lite.env` | MuJoCo, Gymnasium | `VialsLiteEnv`: reset com sorteio ou cena fixa, step a 60 Hz, observações no formato do dataset, estado privilegiado |
| `vials_lite.scripted` | MuJoCo | política roteirizada com IK: baseline e teste de sanidade |
| `vials_lite.lerobot_env` | LeRobot | `EnvConfig` registrado como `vials_lite` para o `lerobot-eval` |
| `evalkit.stats` | NumPy | IC de Wilson, diferença não pareada (Newcombe), diferença pareada, McNemar exato |
| `evalkit.progress` | NumPy | fases do episódio a partir do estado privilegiado (`ProgressTracker`) |
| `evalkit.taxonomy` | NumPy | categoria de falha por regras sobre o resumo do episódio |
| `evalkit.grasps` | NumPy | fechamentos da garra: onde fecha em relação ao frasco, se ele sobe, se a garra o segue |
| `evalkit.policy` | NumPy | protocolos `Policy` e `Probe` |
| `evalkit.lerobot_policy` | LeRobot | `LeRobotPolicy`: `from_pretrained` + `select_action` |
| `evalkit.groot_bridge` | pyzmq, msgpack | servidor e cliente no protocolo ZMQ do GR00T |
| `evalkit.runner` | MuJoCo (+ LeRobot) | o loop de avaliação, as gravações e a CLI |
| `evalkit.report` | NumPy | `report.md` de um run |

## Princípios

1. **Núcleo sem simulador.** `spec`, `joint_mapping`, `success`, `stats`, `progress`, `taxonomy`, `grasps` e `policy` não importam MuJoCo nem PyTorch. São testados no CI sem GPU, em Linux e Windows, e servem para outro simulador: o mesmo `success.py` e o mesmo `progress.py` classificam os episódios gravados no Isaac (`scripts/isaac_rec_convert.py`).
2. **O ambiente não sabe de métricas; o evalkit não sabe de física.** A ponte entre os dois é o estado privilegiado que o ambiente expõe em `info["privileged"]`: poses dos objetos, força de contato da garra com cada frasco e eventos. A função `runner.probe` o converte em `ProgressState`. O protocolo `Probe` existe para backends de outros simuladores.
3. **Compatibilidade antes de elegância.** Nomes, unidades e tempo seguem o dataset do curso, mesmo quando estranhos: o mapeamento afim de juntas, e o "30 fps" que na prática é 1 frame por step de 60 Hz. Assim, uma política treinada no dataset roda sem adaptação.
4. **Fonte única.** Todo número da cena está em `spec.py`, com a origem no comentário; o resto do código importa de lá.
5. **Dados pesados fora do git.** Rollouts, vídeos, cenas reconstruídas, datasets e checkpoints vão para `outputs/`, `datasets/` e `checkpoints/`, ignorados.

## Interfaces principais

```python
# evalkit/policy.py
class Policy(Protocol):
    name: str
    def reset(self) -> None: ...
    def act(self, observation: dict, instruction: str) -> np.ndarray: ...   # (6,), espaço do LeRobot

# vials_lite/env.py
VialsLiteEnv(variant="eval" | "dr-eval", layout="workshop" | "dataset",
             init_pose="workshop" | "dataset", external_cam="random" | "fixed",
             actuators="isaac" | "menagerie", task=..., render_images=True)
env.reset(seed=s)                            # sorteio da variante
env.reset(seed=s, options={"scene": {...}})  # cena fixa (env.SCENE_KEYS), ex. uma cena reconstruída

# vials_lite/success.py
SuccessTracker.update(StepState) -> bool     # True = sucesso confirmado
```

**Observação** (as chaves do dataset do curso): `agent_pos` (6 floats, espaço do LeRobot) e `pixels` com `ego` e `external_D455` (uint8 480×640×3). **Ação:** 6 floats, −100..100 nas juntas do braço e 0..100 na garra. Um step dura 1/60 s.

**Estado privilegiado** (`info["privileged"]`): step, juntas, posição da ponta da garra, poses dos frascos e do rack, força de contato mandíbula ↔ frasco e eventos (`GRASP`, `RELEASE` e `RACK:<frasco>`, quando um frasco conta como colocado).

## Fluxo de uma avaliação

1. `python -m evalkit.runner --policy <pasta ou repo> --variant eval --episodes 100 --seed 1000 --out outputs/evalkit/<nome>`. O padrão é 20 episódios e semente 1000.
2. Antes do primeiro episódio, o runner confere o renderizador e recusa rodar fora da NVIDIA (`--any-renderer` libera).
3. Em cada episódio *i*: `env.reset(seed=seed + i)`, ou a cena *i* de `--scenes`. Seguem 10 steps de aquecimento segurando a pose inicial, como no workshop, e depois `policy.act(obs, instrução)` → `env.step`, até o sucesso ou os 450 steps.
4. A cada step, o runner grava o estado privilegiado, a ação e a fase. No fim do episódio, `progress` dá a fase máxima, `taxonomy` a categoria de falha e `grasps` os fechamentos da garra.
5. No fim do run, `stats` dá o IC de Wilson e `report` escreve o `report.md`.

```
seed ─▶ env.reset ─▶ [aquecimento 10 steps] ─▶ policy.act ─▶ env.step ─┬─▶ sucesso / 450 steps
                                                    ▲                   │
                                                    └── obs ────────────┘
                     info["privileged"] ─▶ probe ─▶ ProgressTracker ─▶ fase por step
fim do episódio ─▶ resumo ─▶ taxonomy.classify + grasps ─▶ episodes.jsonl, steps/ep_NNN.npz
fim do run      ─▶ stats.wilson_interval ─▶ run.json ─▶ report.md
```

## Opções do runner

| Opção | Efeito |
|---|---|
| `--variant {eval,dr-eval}` | variante do workshop |
| `--dr-components dome,robot_color,mat_yaw` | liga só alguns extras da `dr-eval`; os sorteios dos desligados são consumidos, então a cena continua pareada com a `dr-eval` completa |
| `--layout dataset` | lightbox, tapete, rack e frascos no arranjo do dataset (~5 cm mais perto do robô) |
| `--init-pose dataset` | parte da pose mediana dos operadores no dataset |
| `--external-cam fixed` | câmera externa calibrada, sem sorteio |
| `--actuators {isaac,menagerie}` | preset dos atuadores |
| `--instruction` | texto dado à política (padrão: o do curso) |
| `--scenes <scenes.jsonl>` | um episódio por cena fixa (reconstruída do dataset, ou gravada no Isaac) |
| `--ablate vials,joint_pos,...` | com `--scenes`, troca fatores da cena pelos do sorteio do workshop |
| `--replay <dataset ou .rec>` | troca a política pelas ações gravadas, em malha aberta |
| `--policy groot://host:porta` | consulta um servidor no protocolo do GR00T; `--action-horizon` e `--rename-map` como no cliente do workshop |
| `--rep N` | outra semente para a amostragem da política, com as mesmas cenas |
| `--video N` | grava a câmera externa dos N primeiros episódios |
| `--rescore <run>` | reclassifica um run gravado, sem simular |

`--layout`, `--init-pose` e `--external-cam` não mudam os números sorteados: com a mesma `--seed`, episódios de condições diferentes ficam pareados.

## Formatos de saída

```
outputs/evalkit/<run>/
  run.json              política, variante e opções, semente, render (GL_RENDERER),
                        versões (MuJoCo, LeRobot, torch, transformers, commit), taxa e IC
  episodes.jsonl        1 linha por episódio: sucesso, fase máxima, categoria de falha,
                        fechamentos da garra, cena, luz, cor, eventos
  steps/ep_000.npz      arrays por step: juntas, ação, fase, garra, poses, forças de contato
  videos/ep_000.mp4     câmera externa (opcional, --video N)
  report.md             taxa e IC, funil por fase, falhas, fechamentos da garra, tabela por episódio
```

JSONL e NPZ em vez de Parquet: o núcleo não depende de pyarrow.

## Validação contra o Isaac

```
máquina do Isaac                                   ambiente deste repositório
┌───────────────────────────────┐   ZMQ (REQ/REP)  ┌──────────────────────────────┐
│ lerobot_eval.py do workshop   │ ───────────────▶ │ evalkit.groot_bridge         │
│ (sem mudanças), envolvido por │   obs + texto    │  LeRobotPolicy (ACT, pi0.5)  │
│ scripts/isaac_recorder.py     │ ◀─────────────── │  ou ações gravadas (--replay)│
└──────────────┬────────────────┘   chunk de ações └──────────────────────────────┘
               ▼ <variante>.rec (cena inicial, estado por step)
      scripts/isaac_rec_convert.py ─▶ run do evalkit (fases e falhas no Isaac) + scenes.jsonl
      evalkit.runner --scenes ...  ─▶ as mesmas cenas no Vials-Lite
      scripts/paired_compare.py    ─▶ diferença pareada, McNemar, ordem entre políticas
```

O roteiro está em [validacao-isaac.md](validacao-isaac.md).

## Testes

`tests/` tem cerca de 100 funções de teste em 16 arquivos (pytest). Os do núcleo rodam sem MuJoCo e sem GPU; os outros usam `pytest.importorskip` e são pulados com o motivo quando falta o MuJoCo, o Menagerie, o LeRobot, o pyzmq ou o usd-core.

- **Núcleo:** conversão de juntas (ida e volta, limites), critério de sucesso (incluindo a janela efetiva de agarre), estatística (Wilson, Newcombe, pareada, McNemar), fases, taxonomia e fechamentos da garra.
- **Ambiente:** reset e step, determinismo com a mesma semente, cenas fixas, variantes e componentes da DR, layout e pose do dataset, luz, preset dos atuadores, repouso dos frascos no rack e a demo roteirizada completando a tarefa.
- **Integração:** o plugin do `lerobot-eval`, o runner com uma política falsa, a ponte ida e volta por ZMQ, o gravador do Isaac com um ambiente falso no estilo do Isaac Lab, a conversão do registro, a convenção de frames da re-renderização do dataset e a leitura dos USD.

O CI (`.github/workflows/ci.yml`) roda `ruff` e `pytest` em Linux e Windows a cada push, com os extras `dev`, `usd` e `bridge`: o núcleo, a leitura dos USD e a ponte. Os testes de cena e do LeRobot ficam para a máquina com GPU.
