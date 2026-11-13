# Vials-Lite no EnvHub do LeRobot

Este diretório é o conteúdo do repositório de ambiente no Hugging Face Hub. O `lerobot-eval` (LeRobot 0.6.1) baixa o `env.py` e chama `make_env(n_envs, use_async_envs)`. A implementação fica no pacote `vials_lite` (`src/vials_lite/lerobot_env`), que precisa estar instalado na máquina que avalia.

## Uso local, sem Hub (plugin)

```bash
uv sync --locked --extra sim --extra lerobot   # uma vez
MUJOCO_GL=egl uv run --no-sync lerobot-eval \
  --env.discover_packages_path=vials_lite.lerobot_env --env.type=vials_lite \
  --env.variant=eval --policy.path=CAMINHO_DO_CHECKPOINT \
  --eval.n_episodes=10 --eval.batch_size=1 --output_dir=outputs/eval
```

- `CAMINHO_DO_CHECKPOINT`: pasta local ou `usuario/repo` do Hub. Sem checkpoint, `--policy.type=act` cria um ACT com pesos aleatórios (útil para testar o pipeline; sucesso 0%).
- `--no-sync`: sem ele, o `uv run` sincroniza o ambiente e pode remover os extras instalados.
- No PowerShell, tire o `MUJOCO_GL=egl` (no Windows o MuJoCo renderiza por WGL) e troque as quebras `\` por crases.

- `--env.variant`: `eval` (cena do `VialsToRackEvalEnvCfg`) ou `dr-eval` (cor do robô e tapete randomizados).
- `--env.layout`: `workshop` (padrão) ou `dataset` (lightbox, tapete, rack e frascos no arranjo do dataset do curso, ~5 cm mais perto do robô).
- `--env.init_pose`: `workshop` (padrão) ou `dataset` (o braço parte da pose mediana dos operadores no dataset).
- `--env.external_cam`: `random` (padrão, sorteio do workshop) ou `fixed` (a câmera calibrada no vídeo, sem sorteio).
- `--env.actuators`: `isaac` (padrão: o PD do workshop, que reproduz as trajetórias do dataset) ou `menagerie` (os atuadores do modelo do MuJoCo Menagerie).
- `--env.task`: texto da tarefa dado à política. Padrão: o do protocolo do curso, "Pick up the vial and place it in the yellow rack". O dataset do curso usa "Pick up the vial and place it in the rack".

## Uso pelo Hub

Depois de publicado (o dono do repositório faz o upload deste diretório):

```python
from lerobot.envs.factory import make_env
envs = make_env("<usuario>/vials-lite", n_envs=1, trust_remote_code=True)
```

Prefira fixar uma revisão (`"<usuario>/vials-lite@<commit>"`): o `env.py` é código executado na sua máquina.

## Observações e ações

| ambiente | política (`lerobot-eval`) | formato |
|---|---|---|
| `agent_pos` | `observation.state` | float32 (6,), espaço do LeRobot |
| `pixels/ego` | `observation.images.ego` | uint8 480×640×3 |
| `pixels/external_D455` | `observation.images.external_D455` | uint8 480×640×3 |
| ação | `action` | float32 (6,), −100..100 (garra 0..100) |

As chaves são as do dataset do curso (`sreetz-nv/so101_teleop_vials_rack_left`). Um step dura 1/60 s, o episódio tem 450 steps, e o sucesso (`info["is_success"]`) segue o critério do workshop.
