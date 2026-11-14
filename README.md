# litebench-robotics

**A tarefa Vials-to-Rack do [workshop Sim-to-Real SO-101 da NVIDIA](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html), em MuJoCo, rodando num notebook comum, com avaliação diagnóstica de políticas de manipulação.**

O workshop da NVIDIA é a porta de entrada mais completa para o braço SO-101. Ele traz um dataset público de 75 demonstrações, um baseline publicado (GR00T N1.6, 50–70% de sucesso em simulação) e o caminho até o robô real. Mas exige o Isaac Sim, cujo mínimo oficial é uma GPU com 16 GB e RT cores. Quem tem uma placa de 6–8 GB só consegue ler o curso.

Este repositório tem dois pacotes Python:

- **`vials_lite`:** a mesma cena em MuJoCo: robô, frascos, rack, câmeras, randomizações e critério de sucesso. Ela é compatível com o dataset do curso, então as políticas treinadas nele rodam sem adaptação, e funciona como ambiente do `lerobot-eval` (LeRobot 0.6.1). A física roda na CPU e o render é leve, numa GPU de 6–8 GB.
- **`evalkit`:** avaliação de políticas que vai além da taxa de sucesso. Ele mede o progresso por fases a partir do estado do simulador, classifica as falhas automaticamente, analisa os fechamentos da garra e calcula intervalos de confiança. Avalia no sorteio do workshop, nas cenas do dataset reconstruídas pelo vídeo e nas cenas gravadas no próprio Isaac. Uma ponte no protocolo do GR00T leva as mesmas políticas para o avaliador original do workshop, no Isaac.

Para quem: estudantes e pesquisadores que querem seguir o workshop, treinar ou comparar políticas para o SO-101 sem uma GPU de 16 GB, e quem precisa saber *por que* uma política falha, não só *quanto*.

## Resultado principal

O Vials-Lite foi validado contra o Isaac com seis políticas, entre elas os dois checkpoints do GR00T N1.6 publicados pelo curso. Sucesso em 100 episódios por célula, com IC 95% de Wilson, no protocolo do workshop:

| Política | Vials-Lite `eval` | Vials-Lite `dr-eval` | Isaac `eval` | Isaac `dr-eval` |
|---|---|---|---|---|
| ACT do curso | 10% (6–17) | 17% (11–26) | 20% (13–29) | 21% (14–30) |
| pi0.5 | 39% (30–49) | 34% (25–44) | 40% (31–50) | 49% (39–59) |
| GR00T N1.6, simulação + real | 66% (56–75) | 54% (44–63) | 57% (47–66) | 60% (50–69) |
| GR00T N1.6, só simulação | 61% (51–70) | 54% (44–63) | 49% (39–59) | 63% (53–72) |

Com os dois simuladores nas mesmas 100 cenas, a ordem entre as seis políticas avaliadas se mantém (τ de Kendall de +0,69 a +0,87). Repetindo aqui, em malha aberta, as ações gravadas no Isaac, o resultado de cada episódio coincide em 81% dos casos; o próprio Isaac, com as mesmas ações perturbadas em 0,01°, coincide consigo mesmo em 90%. A exceção conhecida são as políticas ACT treinadas com imagens do Isaac, que ficam ~12 pontos abaixo no Vials-Lite em cenas fora da amostra de calibração. Detalhes, protocolo e o resto da validação estão em [docs/resultados.md](docs/resultados.md).

## Requisitos

- Linux ou Windows, Python 3.12 (o [uv](https://docs.astral.sh/uv/) instala) e ~10 GB de disco para o ambiente com o LeRobot.
- **Só o núcleo e os testes:** qualquer máquina, sem GPU.
- **Cena e avaliação:** uma GPU NVIDIA para o render. Desenvolvido e testado numa RTX 5050 de 8 GB (Ubuntu 24.04) e numa RTX 4050 de 6 GB (Windows 11). O pi0.5 ocupa ~6,8 GB de GPU; o ACT, bem menos.
- As RTX 50xx precisam de PyTorch com CUDA ≥ 12.8. O PyTorch do `uv.lock` usa CUDA 13 e precisa de driver NVIDIA ≥ 580.

## Começo rápido

**Linux (bash)**

```bash
git clone https://github.com/WladRamos/litebench-robotics.git && cd litebench-robotics
uv sync --locked --extra dev --extra sim
uv run pytest -q          # sem o Menagerie, os testes de cena são pulados

git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git assets/third_party/mujoco_menagerie
uv run pytest -q          # agora com a cena
MUJOCO_GL=egl uv run python scripts/bench_render.py --menagerie assets/third_party/mujoco_menagerie
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/WladRamos/litebench-robotics.git; cd litebench-robotics
uv sync --locked --extra dev --extra sim
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git assets/third_party/mujoco_menagerie
uv run pytest -q
uv run python scripts/bench_render.py --menagerie assets/third_party/mujoco_menagerie   # sem MUJOCO_GL
```

O `bench_render.py` imprime qual GPU está renderizando. Em notebook com GPU híbrida, confira que é a NVIDIA ([docs/instalacao.md](docs/instalacao.md)).

**Avaliar uma política do Hub** (precisa do extra `lerobot`):

```bash
uv sync --locked --extra dev --extra sim --extra lerobot
MUJOCO_GL=egl uv run python -m evalkit.runner --policy arjunjuneja/act_vials_50k \
    --variant eval --episodes 20 --out outputs/evalkit/act_eval
```

A saída fica em `outputs/evalkit/act_eval/`: `report.md` com a taxa e o IC, o funil por fase, as falhas e os fechamentos da garra, e os dados brutos (`run.json`, `episodes.jsonl`, `steps/*.npz`). No PowerShell, tire o `MUJOCO_GL=egl` e troque as quebras `\` por crases.

## Documentação

| | |
|---|---|
| [Instalação](docs/instalacao.md) | extras, GPU, assets de terceiros, dataset do curso, problemas comuns |
| [Manual de uso](docs/manual.md) | as tarefas passo a passo: instalar, rodar a cena, avaliar, comparar, validar contra o Isaac |
| [Arquitetura](docs/arquitetura.md) | módulos, interfaces, fluxo de uma avaliação, formatos de saída |
| [Especificação da tarefa](docs/especificacao-tarefa.md) | a cena do workshop, número por número, e as aproximações do MuJoCo |
| [Validação contra o Isaac](docs/validacao-isaac.md) | como reproduzir a comparação com o avaliador original |
| [Resultados](docs/resultados.md) | protocolo, políticas, matriz Isaac × Vials-Lite e validação do ambiente |
| [scripts/README.md](scripts/README.md) | o que cada script faz |
| [envhub/README.md](envhub/README.md) | o Vials-Lite pelo `lerobot-eval` |

## Estrutura

```
src/vials_lite/   ambiente: spec, conversão de juntas e sucesso (NumPy puro); cena e
                  ambiente Gymnasium (MuJoCo); demo roteirizada; plugin do lerobot-eval
src/evalkit/      avaliação: estatística, fases, falhas e fechamentos da garra (NumPy puro);
                  runner, relatório, política LeRobot e ponte no protocolo do GR00T
envhub/           o env.py para o EnvHub do LeRobot
scripts/          benchmark, assets, replay e reconstrução do dataset, calibração, Isaac
tests/            pytest; o núcleo roda sem MuJoCo e sem GPU (CI em Linux e Windows)
assets/           malhas visuais do rack e do frasco e instruções para os assets de terceiros
docs/             documentação técnica e de uso
```

## Limitações

- **Simulação só.** Nada aqui foi testado no robô real.
- **Aparência aproximada.** A luz e a câmera externa foram calibradas contra o vídeo do dataset, mas o MuJoCo não faz path tracing. Políticas treinadas com imagens do Isaac perdem alguns pontos aqui (o gap visual, medido em [docs/resultados.md](docs/resultados.md)).
- **Contato aproximado.** O Isaac usa colisores SDF das malhas; aqui o frasco e o rack são montados com primitivas ([docs/especificacao-tarefa.md](docs/especificacao-tarefa.md)). A pega do frasco é a maior fonte de discordância que sobra.
- **Uma tarefa e um robô.** `evalkit.stats`, os protocolos de `evalkit.policy` e a ponte são genéricos; as fases, a taxonomia de falhas e os fechamentos da garra são da tarefa dos frascos, e a conversão de juntas é do SO-101. Uma tarefa nova exigiria uma interface de tarefa com fases, sucesso e regras de falha.
- **Critério de sucesso do workshop.** Ele não verifica se o frasco entrou no furo: em pé sobre o tampo também conta. `scripts/success_seating.py` separa os casos.

## Licença e atribuição

Apache-2.0 ([LICENSE](LICENSE)). Partes derivadas do workshop Sim-to-Real SO-101 da NVIDIA (Apache-2.0) e do MuJoCo Menagerie (Apache-2.0); ver [NOTICE](NOTICE). O dataset do curso (`sreetz-nv/so101_teleop_vials_rack_left`) não declara licença e não é redistribuído aqui: os scripts o leem do Hugging Face Hub.

Para citar:

```bibtex
@software{litebench_robotics,
  author  = {Ramos, Wladimir},
  title   = {litebench-robotics: a tarefa Vials-to-Rack em MuJoCo, com avaliação diagnóstica de políticas},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/WladRamos/litebench-robotics}
}
```
