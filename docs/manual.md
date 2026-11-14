# Manual de uso

Cada tarefa abaixo traz o objetivo, os passos e as exceções: o que pode dar errado e o que fazer. Os comandos são para bash. No PowerShell, tire o `MUJOCO_GL=egl` e troque as quebras de linha `\` por crases. Os comandos mais usados, prontos para copiar, estão em [comandos.md](comandos.md).

## A. Instalar e verificar

**Objetivo:** ter o ambiente pronto e confirmar que a cena roda na GPU certa.

1. Instale o [uv](https://docs.astral.sh/uv/) e clone o repositório.
2. `uv sync --locked --extra dev --extra sim` (acrescente `--extra lerobot` para avaliar políticas).
3. Clone o MuJoCo Menagerie em `assets/third_party/mujoco_menagerie`.
4. `MUJOCO_GL=egl uv run pytest -q`: todos verdes, sem testes de cena pulados.
5. `MUJOCO_GL=egl uv run python scripts/bench_render.py --menagerie assets/third_party/mujoco_menagerie`: confira que o `GL_RENDERER` é NVIDIA e anote os steps por segundo.

**Exceções:**
- Testes de cena pulados com "MuJoCo Menagerie ausente": o passo 3 não foi feito, ou o clone está em outro lugar (defina `VIALS_LITE_MENAGERIE`).
- No Linux sem monitor, um teste de render falha com `an OpenGL platform library has not been loaded`: faltou o `MUJOCO_GL=egl`.
- `GL_RENDERER` com Intel, Mesa ou llvmpipe: o render está fora da NVIDIA. Veja a GPU híbrida em [instalacao.md](instalacao.md#4-gpu-e-render).
- `uv sync --locked` recusa instalar: o `uv.lock` não corresponde ao `pyproject.toml`. Não use `uv lock` para "consertar" sem saber por quê, porque as versões mudam.

Detalhes em [instalacao.md](instalacao.md).

## B. Rodar a cena e a demo roteirizada

**Objetivo:** ver o ambiente funcionando sem nenhuma política treinada.

1. Abra um Python no ambiente (`uv run python`) e rode:

   ```python
   from vials_lite.env import VialsLiteEnv
   from vials_lite.scripted import ScriptedPickPlace

   env = VialsLiteEnv(variant="eval")
   obs, info = env.reset(seed=1)
   print(obs["agent_pos"], obs["pixels"]["ego"].shape)    # juntas (6,) e imagem 480×640×3

   result = ScriptedPickPlace(env).run(seed=1)
   print(result["success"], result["steps"], result["events"])
   ```

2. A demo agarra um frasco, põe em pé sobre um furo livre, solta e recua. Ela deve terminar com sucesso, com os eventos `GRASP`, `RELEASE` e `RACK:<frasco>`.
3. Para assistir no viewer do MuJoCo, episódio após episódio: `uv run python scripts/watch_policy.py --policy scripted` (precisa de monitor). O mesmo comando, com uma política no `--policy`, mostra a política jogando ([comandos.md](comandos.md#ver)).
4. Para salvar imagens, grave `obs["pixels"]["external_D455"]` com qualquer biblioteca de imagem. Para vídeo de um episódio inteiro, use o runner com `--video N` (tarefa C).

**Exceções:**
- `FileNotFoundError: MJCF do Menagerie não encontrado`: tarefa A, passo 3.
- O `watch_policy.py` recusa rodar com `MUJOCO_GL=egl`: o viewer abre uma janela e precisa de monitor. Por SSH, grave vídeo com o runner (`--video N`).
- No Linux sem monitor, erro ao criar o contexto OpenGL: defina `MUJOCO_GL=egl` antes de abrir o Python.
- `render_images=False` no construtor dispensa o render e acelera, quando só a física interessa.

## C. Avaliar uma política

**Objetivo:** a taxa de sucesso de uma política, com IC, funil por fase e falhas.

1. `uv sync --locked --extra dev --extra sim --extra lerobot`.
2. Rode o runner com a política (pasta local ou repositório do Hub):

   ```bash
   MUJOCO_GL=egl uv run python -m evalkit.runner --policy arjunjuneja/act_vials_50k \
       --variant eval --episodes 100 --seed 1000 --video 4 --out outputs/evalkit/act_eval
   ```

3. Abra `outputs/evalkit/act_eval/report.md`: taxa e IC 95%, funil por fase (alcançou o frasco, agarrou, levantou, sobre o rack, soltou no rack, em pé, sucesso), categorias de falha, fechamentos da garra e uma tabela por episódio. Os vídeos ficam em `videos/`.
4. Para ver a política antes ou depois de medir: `uv run python scripts/watch_policy.py --policy arjunjuneja/act_vials_50k`. A semente de cada episódio aparece no terminal, e `--seed <semente> --episodes 1 --video 1` no runner repete aquele episódio com relatório.
5. Para outras condições, troque `--variant dr-eval`, ou use `--layout dataset`, `--init-pose dataset`, `--external-cam fixed` ([arquitetura.md](arquitetura.md#opções-do-runner)).

**Alternativa pelo `lerobot-eval`** (o formato de avaliação do próprio LeRobot):

```bash
MUJOCO_GL=egl uv run --no-sync lerobot-eval \
  --env.discover_packages_path=vials_lite.lerobot_env --env.type=vials_lite \
  --env.variant=eval --policy.path=arjunjuneja/act_vials_50k \
  --eval.n_episodes=10 --eval.batch_size=1 --output_dir=outputs/eval
```

Ele dá a taxa de sucesso, mas não as fases nem as falhas ([envhub/README.md](../envhub/README.md)).

**Exceções:**
- O runner recusa rodar e cita o render: a GPU não é a NVIDIA. Corrija (tarefa A); `--any-renderer` força, mas um episódio leva ~85 s em vez de ~1,7 s.
- `CUDA out of memory` com o pi0.5: ele precisa de ~6,8 GB livres; feche outros processos na GPU.
- Erro 401/403 no pi0.5: tokenizer do PaliGemma restrito ([instalacao.md](instalacao.md#5-políticas-e-dataset-do-curso)).
- Política treinada com outra versão do LeRobot: pode carregar e se comportar diferente. Avalie com a mesma versão do treino.
- Um run interrompido não é retomado: rode de novo com outro `--out`.
- O pi0.5 e outras políticas com amostragem variam ±5–10 pontos entre runs. Para medir essa variação, repita com `--rep 1`, `--rep 2` etc., nas mesmas cenas.

## D. Comparar políticas e ler o relatório

**Objetivo:** saber se uma política é melhor que outra, e onde cada uma falha.

1. Avalie as duas com a mesma `--variant`, `--episodes` e `--seed` (tarefa C): com a mesma semente, os episódios são as mesmas cenas e a comparação é pareada.
2. Compare as taxas pelos IC do `report.md` de cada uma. Para a diferença pareada, use `evalkit.stats`:

   ```python
   import json
   from evalkit.stats import mcnemar_exact, paired_difference

   def outcomes(run):
       return [json.loads(l)["success"] for l in open(f"{run}/episodes.jsonl", encoding="utf-8")]

   a, b = outcomes("outputs/evalkit/pi05_eval"), outcomes("outputs/evalkit/act_eval")
   both = sum(x and y for x, y in zip(a, b)); only_a = sum(x and not y for x, y in zip(a, b))
   only_b = sum(y and not x for x, y in zip(a, b)); neither = len(a) - both - only_a - only_b
   print(paired_difference(both, only_a, only_b, neither), mcnemar_exact(only_a, only_b))
   ```

3. Leia o funil e as falhas de cada `report.md`. Uma política que falha antes de levantar o frasco tem um problema de agarre; uma que leva o frasco e falha no rack tem um problema de colocação. A seção de fechamentos da garra diz se ela fecha longe do frasco, no frasco sem levantar, ou levanta e perde.
4. `scripts/success_seating.py <run>` separa os sucessos em encaixado, no furo, sobre o tampo e pré-colocado.
5. Depois de mudar as regras de fase ou de falha no código, `python -m evalkit.runner --rescore <run>` reclassifica um run sem simular de novo.

**Exceções:**
- Runs com sementes ou números de episódios diferentes não são pareados: use os IC separados (ou `evalkit.stats.newcombe_difference`).
- "Sem diferença" com 100 episódios não quer dizer igual: diferenças de até ~15 pontos passam sem ser detectadas.

## E. Validar contra o Isaac

**Objetivo:** confirmar no avaliador original do workshop um resultado obtido aqui, ou estender a matriz de [resultados.md](resultados.md).

É a tarefa mais pesada: exige uma máquina Linux com o Isaac Sim 5.1, uma GPU de 24 GB com RT cores e o workshop instalado. O roteiro completo, com as exceções conhecidas, está em [validacao-isaac.md](validacao-isaac.md). Em resumo:

1. Na máquina do Isaac, instale este repositório com os extras `lerobot` e `bridge`.
2. Rode a fumaça e depois as células com `scripts/isaac_eval_cells.sh`.
3. Copie `outputs/isaac/` para a máquina do Vials-Lite e converta com `scripts/isaac_rec_convert.py`.
4. Rode as mesmas cenas aqui com `evalkit.runner --scenes`, e as mesmas ações com `--replay`.
5. Compare com `scripts/paired_compare.py`.

## F. Reconstruir as cenas do dataset

**Objetivo:** avaliar as políticas nas cenas em que as demonstrações foram gravadas, ou repetir as ações gravadas em malha aberta.

1. Baixe o dataset do curso ([instalacao.md](instalacao.md#5-políticas-e-dataset-do-curso)).
2. Ajuste a câmera externa de cada episódio: `scripts/fit_external_camera.py` (horas, paralelizável com `--workers`).
3. Reconstrua as cenas: `scripts/reconstruct_scenes.py ... --camera <ajuste> --out outputs/scenes/scenes.jsonl`. Confira as sobreposições com `--overlays`.
4. Avalie: `evalkit.runner --policy <política> --scenes outputs/scenes/scenes.jsonl`.
5. Ações gravadas, sem política: `uv run --with pyarrow python -m evalkit.runner --replay datasets/so101_teleop_vials_rack_left --scenes outputs/scenes/scenes.jsonl`.

As linhas de comando completas, com as dependências extras de cada script, estão nas docstrings e em [scripts/README.md](../scripts/README.md).

**Exceções:**
- `ModuleNotFoundError` de `av`, `pyarrow`, `scipy` ou `pillow`: os scripts declaram essas dependências com `uv run --with ...`, porque elas não fazem parte do pacote.
- A reconstrução pode errar um frasco tapado pelo braço no primeiro frame. O campo `check` da saída diz se a garra, no agarre gravado, cai sobre um frasco reconstruído. Descarte ou corrija os episódios que falham nessa conferência.
