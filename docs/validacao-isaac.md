# Validação contra o Isaac

Este roteiro avalia as mesmas políticas no Isaac, com o avaliador original do workshop, e compara com o Vials-Lite: com sorteio, nas mesmas cenas e com as mesmas ações. Os números obtidos assim estão em [resultados.md](resultados.md).

## Como funciona

O `scripts/lerobot_eval.py` do workshop não carrega políticas: ele consulta um servidor de inferência GR00T por ZMQ. A ponte (`python -m evalkit.groot_bridge`) é esse servidor para uma política do LeRobot 0.6.1. Na máquina do Isaac rodam dois ambientes Python:

| Processo | Ambiente | O que faz |
|---|---|---|
| ponte (`evalkit.groot_bridge`) | o deste repositório (Python 3.12, LeRobot 0.6.1) | carrega a política e responde com o chunk de ações |
| cliente (`lerobot_eval.py`, sem mudanças) | o do Isaac (Python 3.11) | roda a cena, o aquecimento e o critério de sucesso, e conta a taxa |

O gravador (`scripts/isaac_recorder.py`) envolve o cliente sem mudar o arquivo do workshop: troca o `gymnasium.make` por uma versão que registra a cena inicial e o estado de cada step. Ele só lê o estado e não consome números sorteados, então a mesma semente dá os mesmos episódios com e sem gravador.

O que fica igual ao Vials-Lite:
- **Chunk:** a ponte devolve `n_action_steps` ações por consulta (ACT 100, pi0.5 50), e o cliente roda com `--action_horizon 100`. O padrão do cliente, 16, mudaria o protocolo sem dar erro. Por isso a ponte grava as consultas de cada episódio: num episódio de 450 steps que falha, são 5 no ACT e 9 no pi0.5 (com 16, seriam 28).
- **Texto:** "Pick up the vial and place it in the yellow rack", o da página do curso.
- **Câmeras:** as políticas do LeRobot treinadas no dataset do curso esperam `ego` e `external_D455`, os nomes do Isaac, e não precisam de `--rename_map`. O GR00T do curso espera `front` e `wrist`.

## Requisitos

- **GPU:** o Isaac e a política dividem a placa. O pi0.5 ocupa ~6,8 GB com a ponte; o ACT, bem menos. Uma placa de 24 GB com RT cores (RTX 3090, 4090, A5000, L40S, A40) roda tudo. A100 e H100 não servem: não têm RT cores.
- **Driver NVIDIA:** 570–580. O validado é o 580.65.06. O driver 595 derruba o Isaac Sim 5.1 ao iniciar o render RTX (segfault em `librtx.scenedb.plugin`).
- **Isaac:** Isaac Sim 5.1.0, Isaac Lab 2.3.2 e o [workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop) no commit `ce807d9`, instalados como manda o curso. Anote o Python do ambiente do Isaac (`ISAAC_PY`) e o clone do workshop (`WORKSHOP`).
- **Memória:** cada cliente do Isaac ocupa ~12 GB de RAM.

## 1. Preparar este repositório na máquina do Isaac

```bash
git clone https://github.com/WladRamos/litebench-robotics.git && cd litebench-robotics
uv sync --locked --extra lerobot --extra bridge    # o MuJoCo (extra sim) não é necessário aqui
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

O PyTorch do `uv.lock` usa CUDA 13 e precisa de driver ≥ 580. Com driver 570, troque a compilação no `.venv` ([instalacao.md](instalacao.md#4-gpu-e-render)).

Para o pi0.5, o tokenizer `google/paligemma-3b-pt-224` é de acesso restrito: `hf auth login` numa conta que aceitou a licença. Baixe só a raiz do checkpoint (`--exclude "checkpoints/*"`): o repositório inteiro tem ~150 GB.

Se o ambiente do Isaac reclamar de `pyarrow requires NumPy 2.0 or newer`, fixe `"pyarrow==25.0.1"` no Python do Isaac.

## 2. Fumaça (alguns minutos)

```bash
export ISAAC_PY=<python do ambiente do Isaac> WORKSHOP=<clone do workshop>
scripts/isaac_eval_cells.sh fumaca_act_curso arjunjuneja/act_vials_50k 3 Eval
```

Confira em `outputs/isaac/fumaca_act_curso/`:
- `Eval.client.log` termina em `Success Rate: k/3`;
- em `Eval.ponte.log`, a linha `observação` mostra as duas câmeras em `(480, 640, 3)` e o texto com "yellow", e cada episódio tem até 5 consultas;
- `Eval.rec/episodes.jsonl` tem os mesmos sucessos do `client.log`, e a cena de cada episódio não tem `None` nem `_errors` (na `Eval`, só os atributos do domo ficam `None`).

## 3. As células

Uma política por chamada. Cada chamada roda a `Eval` e a `DR-Eval`, com 100 episódios cada:

```bash
scripts/isaac_eval_cells.sh act_curso arjunjuneja/act_vials_50k
scripts/isaac_eval_cells.sh pi05 ltchang/pi05_so101_vials_canonical75_cont_15k
scripts/isaac_eval_cells.sh minha_act <pasta>/pretrained_model
```

Para cada variante, o script:
1. sobe a ponte;
2. roda o cliente pelo gravador, com `PYTHONUNBUFFERED=1` (sem ele, a linha final se perde no buffer);
3. espera a linha `Success Rate`, porque o `lerobot_eval.py` não sai sozinho no fim;
4. encerra os dois.

Variáveis de ambiente do script:

| Variável | Padrão | Efeito |
|---|---|---|
| `PORT` | 5555 | porta da ponte; políticas em paralelo precisam de portas diferentes |
| `SEED` | 1984 | semente do cliente |
| `HORIZON` | 100 | `--action_horizon` do cliente |
| `RECORD` | 1 | grava pelo `isaac_recorder.py`; `0` roda o cliente direto |
| `FRAMES` | 0 | grava vídeo das câmeras nos N primeiros episódios |
| `GROOT_DIR` | — | clone do Isaac-GR00T: sobe o `run_gr00t_server.py` no lugar da ponte |
| `RENAME_MAP` | — | `--rename_map` do cliente |
| `REPLAY` | 0 | `1`: a ponte devolve as ações de uma avaliação gravada (seção 6) |
| `NOISE` | 0 | com `REPLAY=1`, ruído de σ graus nas ações, semeado por episódio |
| `VIAL_SDF` | — | resolução do SDF de colisão dos frascos (o workshop usa 20); exige o gravador |
| `EXTERNAL` | 0 | `1`: não sobe servidor e usa o que já estiver ouvindo em `PORT` |

Saídas em `outputs/isaac/<nome>/`:

| Arquivo | Conteúdo |
|---|---|
| `meta.txt` | data, GPU e driver, commits deste repositório e do workshop, parâmetros |
| `<variante>.client.log` | saída do `lerobot_eval.py`, com a taxa |
| `<variante>.ponte.log`, `.ponte.jsonl` | consultas por episódio (conferência do chunk) |
| `<variante>.rec/` | `meta.json`, `episodes.jsonl` (sucesso, steps e cena inicial de cada episódio) e `ep_NNNN.npz` (ação, juntas, poses, força mandíbula ↔ frascos, por step); ~140 kB por episódio |

**GR00T do curso:** com `GROOT_DIR` apontando para um clone do [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) no commit `ead5283` (com o próprio `.venv`), o script sobe o servidor do GR00T. Use o protocolo do curso:

```bash
GROOT_DIR=<Isaac-GR00T> HORIZON=16 RENAME_MAP='{"external_D455": "front", "ego": "wrist"}' \
  scripts/isaac_eval_cells.sh groot_sim_real <pasta do checkpoint-10000>
```

**Paralelo:** dá para rodar várias políticas ao mesmo tempo, cada uma com o seu `PORT`. Numa GPU de 24 GB, cabem 2 clientes (3 estouram a memória e o PhysX cai com "Scene state is corrupted"); numa A40 de 48 GB, 3. O tempo é de ~20–60 s por episódio de 450 steps, conforme a GPU e o número de clientes.

## 4. Converter o registro para o evalkit

De volta à máquina do Vials-Lite (copie `outputs/isaac/` para lá):

```bash
for p in act_curso pi05; do for v in Eval DR-Eval; do
  uv run python scripts/isaac_rec_convert.py outputs/isaac/$p/$v.rec --name $p \
    --run outputs/evalkit/isaac/${p}_$v --scenes outputs/isaac/$p/$v.scenes.jsonl
done; done
```

Para cada célula, isso gera:
- um run do evalkit com fases, falhas e fechamentos da garra calculados sobre o estado do Isaac. A ponta da garra sai da cinemática do MJCF com as juntas do Isaac, que já foi conferida contra o USD;
- um `scenes.jsonl` com a cena inicial de cada episódio, para o Vials-Lite.

O episódio *i* recebe a semente 1000 + *i*, a mesma que o runner dá à cena *i* de `--scenes`.

## 5. As mesmas cenas no Vials-Lite

```bash
uv run python -m evalkit.runner --policy arjunjuneja/act_vials_50k \
    --scenes outputs/isaac/act_curso/Eval.scenes.jsonl --out outputs/evalkit/d33_isaac_scenes/act_curso_Eval
uv run python -m evalkit.runner --policy arjunjuneja/act_vials_50k \
    --scenes outputs/isaac/act_curso/DR-Eval.scenes.jsonl --out outputs/evalkit/d33_isaac_scenes/act_curso_DR-Eval
```

As duas rodam com a variante padrão (`eval`), inclusive as cenas da `DR-Eval`: as cenas trazem a pose dos objetos, as juntas, a cor do robô, a luz e o giro do tapete do Isaac. A câmera externa recebe o mesmo desvio que o `camera_mount` teve no Isaac, aplicado à câmera calibrada daqui. O HDRI do domo não tem equivalente: o domo daqui é uniforme.

## 6. As mesmas ações

**No Vials-Lite**, as ações do Isaac em malha aberta, na mesma cena:

```bash
uv run python -m evalkit.runner --replay outputs/isaac/pi05/Eval.rec \
    --scenes outputs/isaac/pi05/Eval.scenes.jsonl --out outputs/evalkit/replay_isaac_pi05_Eval
```

**No próprio Isaac**, para medir quanto ele concorda consigo mesmo (o teto da concordância):

```bash
REPLAY=1 SEED=2026 NOISE=0.01 scripts/isaac_eval_cells.sh rpn_pi05_s2026 outputs/isaac/pi05_s2026
```

Use a mesma `SEED` e o mesmo número de episódios da gravação. Com as ações exatas, o PhysX repete a trajetória na mesma máquina, por isso o teto precisa do ruído. Em cada consulta, a ponte confere as juntas recebidas contra as gravadas (`dev_first_deg` e `dev_max_deg` no `.ponte.jsonl`); no replay exato, `dev_first_deg` deve ficar perto de 0.

## 7. Comparação pareada

```bash
uv run python scripts/paired_compare.py --root outputs/evalkit
uv run python scripts/success_seating.py outputs/evalkit/isaac/pi05_Eval outputs/evalkit/d33_isaac_scenes/pi05_Eval
```

O `paired_compare.py` procura os runs do Isaac em `<root>/isaac/<política>_<variante>` e os do Vials-Lite em `<root>/d33_isaac_scenes/<política>_<variante>` (o prefixo muda com `--physics`). Para cada par, ele imprime:
- a tabela 2×2, o IC 95% pareado e o McNemar exato, no critério do workshop e no "entrou no furo";
- a ordem entre as políticas em cada simulador.

O `success_seating.py` diz onde o frasco termina em cada sucesso.

## Conferir a ponte sem o Isaac

Na máquina do Vials-Lite, em dois terminais:

```bash
# terminal 1: a ponte
uv run python -m evalkit.groot_bridge --policy arjunjuneja/act_vials_50k
# terminal 2: o runner, que fala o mesmo protocolo do cliente do workshop
uv run python -m evalkit.runner --policy groot://localhost:5555 --action-horizon 100 --variant eval --episodes 5
```

O mesmo ACT, direto e pela ponte, dá os mesmos episódios. O `scripts/isaac_eval_cells.sh` é só para Linux; a ponte e o runner funcionam também no Windows.

## Cuidados

- **O domo da `DR-Eval` no Isaac não segue a semente:** a textura e a orientação mudam entre execuções (83 de 100 episódios), embora frascos, pose, cores e câmeras se repitam. Para comparar, use a cena gravada da execução correspondente.
- **O Isaac na GPU não repete os episódios** com a mesma semente em malha fechada. Duas rodadas da mesma célula variam alguns pontos; registre as duas lado a lado.
- **Entre máquinas, o replay exato não repete a gravação:** as juntas batem até ~1 s e depois se afastam. O sucesso costuma bater, mas a referência do teto deve ser da mesma máquina.
- O cliente do workshop só devolve a taxa de sucesso. Fases e falhas no Isaac vêm do gravador e da conversão (seção 4).
