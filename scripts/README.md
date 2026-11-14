# scripts/

Ferramentas de linha de comando fora dos pacotes. Cada script tem na docstring o método, as entradas e a linha de comando completa, com as dependências extras (`uv run --with av --with pyarrow ...`) quando ele lê o dataset do curso. No Linux, os que renderizam pedem `MUJOCO_GL=egl`.

Os scripts de calibração ficam nesta pasta, junto com os outros, porque há imports entre eles e os testes os carregam pelo caminho.

## Uso

| Script | O que faz |
|---|---|
| `watch_policy.py` | assiste a uma política (ou à demo roteirizada) no viewer do MuJoCo, em episódios consecutivos até fechar a janela; com `--stream`, no navegador, para máquinas remotas |
| `bench_render.py` | mede quantos steps por segundo a máquina faz, com e sem as duas câmeras, e mostra qual GPU está renderizando |
| `success_seating.py` | classifica os sucessos de um run: encaixado, no furo, sobre o tampo ou só o frasco pré-colocado |

A avaliação em si não é um script: é o `python -m evalkit.runner` ([docs/manual.md](../docs/manual.md)).

## Reprodução a partir do dataset do curso

| Script | O que faz |
|---|---|
| `replay_dataset.py` | replay cinemático dos 75 episódios no MJCF: faixas das juntas e posição da garra no agarre e na soltura |
| `replay_dynamic.py` | replay dinâmico: as ações gravadas passam pelos atuadores simulados e o estado é comparado com o gravado |
| `reconstruct_scenes.py` | reconstrói pelo vídeo a cena inicial de cada episódio (rack, frascos, tapete, luz, cor do robô) → `scenes.jsonl` para o `--scenes` do runner |
| `render_dataset.py` | re-renderiza o dataset no Vials-Lite (mesmas ações, imagens e juntas daqui), como dataset LeRobot v3, para treino |
| `offline_gap.py` | teste offline do gap visual: a política recebe o mesmo estado com imagens do Isaac ou do Vials-Lite, e o chunk previsto é comparado com as ações gravadas |

Ordem para as cenas: `fit_external_camera.py` (calibração, abaixo) → `reconstruct_scenes.py` → `evalkit.runner --scenes` ou `render_dataset.py` ou `offline_gap.py`.

## Validação contra o Isaac

| Script | O que faz |
|---|---|
| `isaac_eval_cells.sh` | na máquina do Isaac: sobe a ponte (ou o servidor do GR00T), roda o `lerobot_eval.py` do workshop e grava as saídas de cada célula |
| `isaac_recorder.py` | gravador: roda o `lerobot_eval.py` sem mudanças, registrando a cena inicial e o estado por step de cada episódio |
| `isaac_rec_convert.py` | converte o registro em um run do evalkit (fases e falhas no Isaac) e em `scenes.jsonl` para o Vials-Lite |
| `paired_compare.py` | Isaac × Vials-Lite nas mesmas cenas: diferença pareada, McNemar, critério "entrou no furo" e ordem entre políticas |

Roteiro em [docs/validacao-isaac.md](../docs/validacao-isaac.md).

## Assets e calibração

Estes regeneram o que já está versionado ou em `spec.py`. Só são necessários para mudar a cena.

| Script | O que faz |
|---|---|
| `inspect_workshop_assets.py` | extrai dos USD do workshop poses de câmeras, furos do rack, extents e malhas (precisa do extra `usd`) |
| `export_workshop_meshes.py` | exporta as malhas visuais do rack e do frasco para `assets/vials_lite/*.obj` |
| `bake_ambient_occlusion.py` | grava a oclusão de ambiente nas malhas visuais (rodar depois do `export_workshop_meshes.py`) |
| `check_alignment.py` | confere a cinemática do USD do robô contra a do MJCF do Menagerie e dá a pose da base e os offsets das juntas |
| `fit_external_camera.py` | ajusta, por episódio, a pose e o FOV da câmera externa ao vídeo do dataset, pela silhueta do robô |
| `measure_appearance.py` | mede a cor do piso, das paredes e do tapete no vídeo e no render |
| `fit_lighting.py` | calibra as constantes de luz (`spec.LightingCalibration`) contra as cores do vídeo |
