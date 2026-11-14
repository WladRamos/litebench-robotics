# Comandos úteis

Os comandos do dia a dia, para copiar. Eles supõem a instalação completa ([instalacao.md](instalacao.md)), com o Menagerie em `assets/third_party/mujoco_menagerie` e os extras `dev`, `sim` e `lerobot`, e partem da raiz do repositório. Os passos e as exceções de cada tarefa estão no [manual](manual.md).

No Linux sem monitor (SSH, servidor), prefixe com `MUJOCO_GL=egl` tudo o que renderiza, menos o viewer, que precisa de monitor (para ver de longe, use o `--stream`, abaixo). No PowerShell, troque as quebras `\` por crases.

## Ver

```bash
# a cena com a demo roteirizada, episódio após episódio (não precisa de política nem do LeRobot)
uv run python scripts/watch_policy.py --policy scripted

# uma política em cenas sorteadas consecutivas, até fechar a janela (ou Ctrl+C)
uv run python scripts/watch_policy.py --policy arjunjuneja/act_vials_50k

# o mesmo na dr-eval, em câmera lenta, a partir de uma semente conhecida
uv run python scripts/watch_policy.py --policy arjunjuneja/act_vials_50k --variant dr-eval --speed 0.5 --seed 1000
```

O terminal mostra, por episódio, a semente, o resultado, a fase máxima e a categoria de falha. O viewer abre na câmera externa, a que a política vê; `--camera ego` ou `--camera free` abrem em outra. Durante a execução, a lista **Camera** do painel esquerdo (seção Rendering) tem todas as câmeras do ambiente, e `[` e `]` percorrem a mesma lista. Esc solta a câmera (arrastar gira, botão direito desloca, roda aproxima) e a barra de espaço pausa.

Num notebook com GPU híbrida no Linux, o viewer pode abrir na integrada: veja [instalacao.md](instalacao.md#4-gpu-e-render).

**De outra máquina** (a política roda num computador remoto, e VNC ou `ssh -X` deixam o viewer travado): a máquina remota renderiza sem janela e serve as imagens, que você abre no navegador por um túnel SSH.

```bash
# na máquina remota (por SSH)
uv run python scripts/watch_policy.py --policy arjunjuneja/act_vials_50k --stream 8000

# na sua máquina: abra o túnel e acesse http://localhost:8000
ssh -N -L 8000:localhost:8000 usuario@maquina-remota
```

A página escolhe a câmera (as do ambiente ou uma livre, que gira arrastando e aproxima com a roda), pausa e lista o resultado de cada episódio. O servidor só aceita conexões da própria máquina; por isso o túnel. Ctrl+C no terminal remoto encerra. Com `--policy scripted` sem o extra `lerobot`, rode com `uv run --with pillow`.

`--policy` aceita uma pasta local com o checkpoint, um repositório do Hub ou `groot://host:porta` (com `--action-horizon`), como no runner.

## Avaliar

```bash
# 100 episódios no protocolo do workshop, com vídeo dos 4 primeiros
uv run python -m evalkit.runner --policy arjunjuneja/act_vials_50k \
    --variant eval --episodes 100 --seed 1000 --video 4 --out outputs/evalkit/act_eval

# a mesma política na variante randomizada
uv run python -m evalkit.runner --policy arjunjuneja/act_vials_50k \
    --variant dr-eval --episodes 100 --seed 1000 --out outputs/evalkit/act_dr-eval

# repetir com relatório e vídeo um episódio visto no viewer (a semente que o terminal mostrou)
uv run python -m evalkit.runner --policy arjunjuneja/act_vials_50k --seed 1234 --episodes 1 --video 1

# pelo lerobot-eval (só a taxa de sucesso)
uv run --no-sync lerobot-eval --env.discover_packages_path=vials_lite.lerobot_env --env.type=vials_lite \
    --env.variant=eval --policy.path=arjunjuneja/act_vials_50k \
    --eval.n_episodes=10 --eval.batch_size=1 --output_dir=outputs/eval
```

Em cada run, `report.md` traz a taxa com o IC 95%, o funil por fase, as falhas e os fechamentos da garra. Os vídeos ficam em `videos/`.

## Analisar um run

```bash
# onde o frasco termina nos sucessos: encaixado, no furo, sobre o tampo ou pré-colocado
uv run python scripts/success_seating.py outputs/evalkit/act_eval

# reclassificar um run gravado depois de mudar as regras de fase ou de falha, sem simular de novo
uv run python -m evalkit.runner --rescore outputs/evalkit/act_eval
```

Para comparar duas políticas avaliadas com a mesma `--seed`, use a diferença pareada da tarefa D do [manual](manual.md#d-comparar-políticas-e-ler-o-relatório).

## Verificar a instalação

```bash
uv run pytest -q                       # nenhum teste de cena pulado se o Menagerie estiver no lugar
uv run python scripts/bench_render.py --menagerie assets/third_party/mujoco_menagerie   # GPU e steps/s
uv run ruff check .
```
