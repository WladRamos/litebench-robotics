# Resultados

A pergunta deste documento: o Vials-Lite reproduz o desempenho que as políticas têm no Isaac, a ponto de substituí-lo na avaliação? Os números abaixo são os da versão atual da física (a de `spec.py` e `scene.py` neste repositório). Cada célula traz o número de sucessos e o IC 95%.

## 1. Protocolo

- **Variantes:** `eval` e `dr-eval`, as do código do workshop. A `dr-eval` tem o mesmo sorteio físico da `eval` e acrescenta cor do robô, luz do domo e mais giro do tapete.
- **Episódios:** 100 por célula. No Vials-Lite, com sorteio, sementes 1000 a 1099.
- **Episódio:** 10 steps de aquecimento segurando a pose inicial, como no workshop. Depois a política age em malha fechada, até 450 steps (7,5 s; 1 step = 1/60 s).
- **Sucesso:** o critério do workshop, portado em `vials_lite.success`: um frasco agarrado, solto em pé dentro do rack e mantido assim por 25 steps seguidos ([especificação](especificacao-tarefa.md#critério-de-sucesso-mdptermspy)). O critério não verifica o furo; a seção 6 separa os casos.
- **Instrução:** "Pick up the vial and place it in the yellow rack", a do protocolo do curso. Só o pi0.5 e o GR00T a usam.
- **Chunk executado por consulta:** ACT 100 ações, pi0.5 50, que é a fila do `select_action` do LeRobot. No Isaac, as políticas do LeRobot passam pela ponte com `--action_horizon 100`, o que dá a mesma execução. O GR00T roda com chunk de 16 nos dois simuladores, como no workshop.
- **Software:** LeRobot 0.6.1 nas políticas do LeRobot. No Isaac: Isaac Sim 5.1.0, Isaac Lab 2.3.2 e o workshop no commit `ce807d9`, em GPUs NVIDIA A40 e RTX 3090.
- **Estatística** (`evalkit.stats`):
  - IC 95% de Wilson para cada taxa;
  - entre simuladores com sorteio, a diferença é **não pareada** (Newcombe), porque a mesma semente não gera a mesma cena no Isaac e aqui;
  - nas mesmas cenas, a diferença é **pareada** (Newcombe para pares), com McNemar exato.

## 2. Políticas

| Política | O que é | Treino | Onde está |
|---|---|---|---|
| **ACT do curso** | ACT (~50M parâmetros), imitação | dataset do curso e episódios extras gerados com Cosmos | Hub, `arjunjuneja/act_vials_50k` |
| **pi0.5** | VLA (~3B), afinado por imitação | `lerobot/pi05_base` afinado nos 75 episódios do curso | Hub, `ltchang/pi05_so101_vials_canonical75_cont_15k` |
| **ACT-Isaac** | a receita do ACT, treinada para a validação cruzada | 66 episódios do curso, com as imagens do Isaac; 50k passos, semente 1000, LeRobot 0.6.1 | não publicado |
| **ACT-Lite** | idem | os mesmos 66 episódios, re-renderizados no Vials-Lite (`scripts/render_dataset.py`) | não publicado |
| **GR00T N1.6, simulação + real** | VLA da NVIDIA (~3B), checkpoint do curso | 75 episódios de simulação + 5 reais; 10.000 passos | Hub, `aravindhs-NV/grootn16-finetune_sreetz-so101_teleop_vials_rack_left_sim_and_real` |
| **GR00T N1.6, só simulação** | idem | os 75 episódios do curso | Hub, `aravindhs-NV/grootn16-finetune_sreetz-so101_teleop_vials_rack_left_sim_only` |

O GR00T é servido pelo `run_gr00t_server.py` do [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (commit `ead5283`), com as câmeras renomeadas para `front` e `wrist`, como no workshop. No Vials-Lite, o runner o consulta com `--policy groot://`.

ACT-Isaac e ACT-Lite diferem só nas imagens de treino. Ficaram fora do treino os 9 episódios (6, 23, 24, 39, 40, 45, 51, 55, 57) em que o replay das ações gravadas não completava a tarefa no Vials-Lite no momento da re-renderização: neles, as imagens daqui mostrariam o frasco escorregando enquanto as ações seguem a demonstração.

## 3. Matriz com sorteio: Isaac × Vials-Lite

Sucesso em 100 episódios (IC 95% de Wilson):

| Política | Vials-Lite `eval` | Vials-Lite `dr-eval` | Isaac `eval` | Isaac `dr-eval` |
|---|---|---|---|---|
| ACT do curso | 10/100 (6–17%) | 17/100 (11–26%) | 20/100 (13–29%) | 21/100 (14–30%) |
| pi0.5 | 39/100 (30–49%) | 34/100 (25–44%) | 40/100 (31–50%) | 49/100 (39–59%) |
| ACT-Isaac | 11/100 (6–19%) | 5/100 (2–11%) | 24/100 (17–33%) | 20/100 (13–29%) |
| ACT-Lite | 15/100 (9–23%) | 16/100 (10–24%) | 4/100 (2–10%) | 16/100 (10–24%) |
| GR00T, simulação + real | 66/100 (56–75%) | 54/100 (44–63%) | 57/100 (47–66%) | 60/100 (50–69%) |
| GR00T, só simulação | 61/100 (51–70%) | 54/100 (44–63%) | 49/100 (39–59%) | 63/100 (53–72%) |

Nas políticas do LeRobot, o Isaac foi rodado duas vezes com a mesma semente (1984). A tabela mostra a primeira rodada; a segunda deu, na `eval` e na `dr-eval`: ACT do curso 19 e 17, pi0.5 34 e 45, ACT-Isaac 16 e 14, ACT-Lite 7 e 17. As diferenças entre as rodadas, de −8 a +3 pontos, ficam dentro do ruído: com a mesma semente, o Isaac na GPU não repete os episódios.

**Diferença Isaac − Vials-Lite** (não pareada; no Isaac, as duas rodadas somadas, 200 episódios; no GR00T, 100), em pontos, `eval` e `dr-eval`:

| Política | `eval` | `dr-eval` |
|---|---|---|
| ACT do curso | **+10 [+1; +17]** | +2 [−8; +11] |
| pi0.5 | −2 [−14; +9] | **+13 [+1; +24]** |
| ACT-Isaac | +9 [0; +17] | **+12 [+4; +18]** |
| ACT-Lite | **−10 [−18; −2]** | +1 [−9; +9] |
| GR00T, simulação + real | −9 [−22; +4] | +6 [−8; +19] |
| GR00T, só simulação | −12 [−25; +2] | +9 [−5; +22] |

Em negrito, os intervalos que não contêm o zero: 5 de 12. A seção seguinte compara nas mesmas cenas, o que tira a variação do sorteio.

## 4. Mesmas cenas: comparação pareada

O gravador (`scripts/isaac_recorder.py`) registra a cena inicial e o estado de cada step dos episódios do Isaac. As mesmas 100 cenas por variante rodam então no Vials-Lite (`evalkit.runner --scenes`), e a diferença é pareada episódio a episódio ([validacao-isaac.md](validacao-isaac.md)).

Duas amostras de cenas:
- **Semente 1984** (a segunda rodada do Isaac): os parâmetros de contato do Vials-Lite foram escolhidos olhando estas cenas.
- **Semente 2026:** fora da amostra, nenhum parâmetro foi ajustado nelas.

O pi0.5 e o GR00T amostram na inferência, e um mesmo run repetido varia ±5–10 pontos (o render da NVIDIA muda um pouco a cada `reset`, e isso basta para mudar a trajetória). Por isso eles têm dois ou três runs por célula no Vials-Lite.

### Semente 1984

Sucessos, Isaac / Vials-Lite (p do McNemar exato):

| Política | `eval` | `dr-eval` |
|---|---|---|
| ACT do curso | 19 / 13 (0,24) | 17 / 12 (0,33) |
| pi0.5 (2 runs) | 34 / 48 (0,049); 43 (0,22) | 45 / 38 (0,41); 45 (1,00) |
| ACT-Isaac | 16 / 6 (0,031) | 14 / 8 (0,26) |
| ACT-Lite | 7 / 13 (0,24) | 17 / 14 (0,63) |
| GR00T, simulação + real (2 runs) | 57 / 64 (0,35); 66 (0,16) | 60 / 62 (0,88); 61 (1,00) |
| GR00T, só simulação (2 runs) | 49 / 67 (0,010); 60 (0,13) | 63 / 45 (0,011); 49 (0,059) |

No primeiro run, 4 das 12 células diferem com p < 0,05. Três delas não se repetem no segundo run (a do pi0.5 na `eval` e as duas do GR00T só simulação, que vão em direções opostas nas duas variantes): são o ruído entre runs. Sobra o ACT-Isaac na `eval`.

IC 95% pareado da diferença Isaac − Vials-Lite (1º run; 2º run entre parênteses):

| Política | `eval` | `dr-eval` |
|---|---|---|
| ACT do curso | +6 [−3; +15] | +5 [−3; +14] |
| pi0.5 | −14 [−26; −1] (−9 [−21; +4]) | +7 [−7; +21] (0 [−13; +13]) |
| ACT-Isaac | +10 [+2; +19] | +6 [−3; +15] |
| ACT-Lite | −6 [−15; +3] | +3 [−5; +12] |
| GR00T, simulação + real | −7 [−19; +6] | −2 [−15; +11] |
| GR00T, só simulação | −18 [−30; −5] (−11 [−23; +2]) | +18 [+5; +30] (+14 [0; +27]) |

"Sem diferença" tem um limite de resolução: com 100 episódios, uma diferença de até ~15 pontos nas ACT, e de até ~25–30 no pi0.5 e no GR00T, passa sem ser detectada.

A concordância episódio a episódio com o Isaac é de 80–83% nas ACT, 47–56% no pi0.5 e 54–59% no GR00T. A amostragem do pi0.5 e do GR00T torna cada episódio pouco repetível mesmo no próprio simulador.

### Semente 2026 (fora da amostra)

Sucessos, Isaac / Vials-Lite (p do McNemar exato):

| Política | `eval` | `dr-eval` |
|---|---|---|
| ACT do curso | 24 / 11 (**0,004**) | 29 / 17 (0,050) |
| pi0.5 (3 runs) | 37 / 41, 39, 40 (0,64; 0,88; 0,75) | 47 / 34, 38, 42 (**0,047**; 0,20; 0,49) |
| GR00T, simulação + real (2 runs) | 59 / 64, 63 (0,52; 0,64) | 54 / 62, 63 (0,30; 0,22) |
| GR00T, só simulação (2 runs) | 58 / 61, 50 (0,77; 0,29) | 59 / 47, 46 (0,11; 0,098) |
| ACT-Isaac | 19 / 8 (**0,019**) | 18 / 7 (**0,027**) |
| ACT-Lite | 11 / 10 (1,00) | 14 / 9 (0,36) |

IC 95% pareado da diferença Isaac − Vials-Lite (1º run):

| Política | `eval` | `dr-eval` |
|---|---|---|
| ACT do curso | **+13 [+5; +22]** | **+12 [+1; +23]** |
| pi0.5 | −4 [−17; +9] | +13 [+1; +24] |
| GR00T, simulação + real | −5 [−17; +7] | −8 [−21; +5] |
| GR00T, só simulação | −3 [−16; +10] | +12 [−2; +25] |
| ACT-Isaac | **+11 [+2; +20]** | **+11 [+2; +20]** |
| ACT-Lite | +1 [−8; +10] | +5 [−4; +14] |

Leitura:
- **pi0.5 e GR00T se sustentam fora da amostra.** No pi0.5, 5 dos 6 runs não diferem do Isaac. Nos dois GR00T, nenhum dos 8 runs difere.
- **As duas ACT treinadas com imagens do Isaac ficam ~12 pontos abaixo aqui**: o ACT do curso e o ACT-Isaac. O ACT do curso foi rodado duas vezes no Isaac, em GPUs diferentes, e a segunda rodada repete a primeira (concordância de 80–85% episódio a episódio). A diferença está no agarre: mais episódios em que a garra toca o frasco e não o levanta.
- **A ACT-Lite, treinada com as imagens do Vials-Lite, quase não perde** (+1 e +5). Como as três ACT compartilham a receita e diferem só nas imagens, a parte da diferença que acompanha as imagens de treino é o gap visual. Nenhuma diferença entre a ACT-Isaac e a ACT-Lite dentro de um mesmo simulador tem p < 0,05; a direção é a mesma das cenas de 1984.

### Ordem entre as políticas

Nos dois simuladores, nas duas variantes e nas duas amostras de cenas, os dois GR00T ficam à frente do pi0.5, que fica à frente do ACT do curso. τ de Kendall entre as ordens das seis políticas no Isaac e no Vials-Lite:

| Cenas | `eval` | `dr-eval` |
|---|---|---|
| semente 1984 | +0,69 | +0,83 |
| semente 2026 | +0,87 | +0,73 |

As trocas são dentro de pares próximos: os dois GR00T entre si na `dr-eval`, e a ACT-Isaac com a ACT-Lite.

pi0.5 − ACT do curso, pareado, nas cenas de 1984 e de 2026 (`eval` e `dr-eval`):
- Isaac: +15 [+3; +26] e +28 [+15; +40]; +13 [0; +25] e +18 [+4; +31];
- Vials-Lite: +35 [+23; +45] e +26 [+15; +36]; +30 [+18; +41] e +17 [+5; +28].

Na `eval`, a margem é maior no Vials-Lite porque o ACT do curso cai.

## 5. Mesmas ações: a física isolada

O runner também repete em malha aberta as ações gravadas no Isaac, na mesma cena (`--replay <variante>.rec`). Sem a política no meio, a diferença de resultado vem só da física.

- Ações do pi0.5 nas cenas de 1984: o Vials-Lite dá 34 e 42 sucessos, contra 34 e 45 no Isaac (`eval` e `dr-eval`).
- O critério de sucesso do Vials-Lite, aplicado ao estado gravado no Isaac, dá o mesmo veredito em 799 de 800 episódios.
- **O teto de concordância.** Repetindo no próprio Isaac as ações gravadas, com uma perturbação de 0,01° (pi0.5 e ACT do curso, `eval`, semente 2026, 200 episódios), a classe final do episódio (sucesso encaixado, no furo, sobre o tampo, pré-colocado ou falha) coincide em 90%. O Vials-Lite coincide com o Isaac em 81% e consigo mesmo em 93%, com a mesma perturbação.
- A discordância que sobra está na pega: no Isaac, o frasco sai mais inclinado da garra, e parte dos sucessos dele em pé sobre o tampo aqui tomba.

## 6. Onde o frasco termina nos sucessos

`scripts/success_seating.py` classifica cada sucesso pela altura do frasco no frame do rack no último step: encaixado (< 0,03 m), no furo (0,03–0,08), sobre o tampo (≥ 0,08) ou só o frasco que já começou no rack. Mesmas cenas, semente 1984; políticas do LeRobot com as 8 células somadas, GR00T com as 4 (1º run):

| | sucessos | encaixado | no furo | sobre o tampo | só o pré-colocado |
|---|---|---|---|---|---|
| Isaac, LeRobot | 169 | 27 (16%) | 94 (56%) | 34 (20%) | 14 (8%) |
| Vials-Lite, LeRobot | 152 | 12 (8%) | 106 (70%) | 20 (13%) | 14 (9%) |
| Isaac, GR00T | 229 | 84 (37%) | 63 (28%) | 82 (36%) | 0 |
| Vials-Lite, GR00T | 238 | 62 (26%) | 59 (25%) | 117 (49%) | 0 |

Nas cenas de 2026, Isaac / Vials-Lite: LeRobot 199 / 137 sucessos, sobre o tampo 25% / 32%; GR00T 230 / 234, sobre o tampo 37% / 51%.

**No critério "entrou no furo"** (sucesso com o frasco encaixado ou no furo), a diferença pareada Isaac − Vials-Lite fica dentro do ruído para o pi0.5 e para o GR00T na `eval`. O Vials-Lite subestima:
- as ACT treinadas com imagens do Isaac: o ACT do curso fora da amostra, +14 [+5; +23] e +11 [+2; +20]; o ACT-Isaac, +9 [+2; +17] na `eval` de 1984 e +10 nas duas variantes de 2026;
- o GR00T na `dr-eval`, nos dois checkpoints: de +14 a +21 pontos.

O GR00T termina mais vezes sobre o tampo aqui (49% contra 36%). Cite os números do GR00T no critério do furo na `dr-eval` com essa ressalva.

## 7. Cenas do dataset

`scripts/reconstruct_scenes.py` reconstrói, pelo vídeo, a cena inicial de cada um dos 75 episódios do dataset do curso: rack, frascos, tapete, câmera, luz e cor do robô. Rodar as políticas nessas cenas (`--scenes`) mede o desempenho perto das condições de treino. Sucesso em 75 episódios:

| Cenas | ACT do curso | pi0.5 | GR00T simulação + real | GR00T só simulação |
|---|---|---|---|---|
| cenas do dataset | 35/75 (36–58%) | 48/75 (53–74%) | 53/75 (60–80%) | 49/75 (54–75%) |
| com a pose inicial do workshop | 19/75 (17–36%) | 26/75 (25–46%) | 32/75 (32–54%) | 32/75 (32–54%) |
| com as câmeras da `eval` | 23/75 (21–42%) | 45/75 (49–70%) | 53/75 (60–80%) | 54/75 (61–81%) |
| com a pose mediana dos operadores | 34/75 (35–57%) | 40/75 (42–64%) | 59/75 (68–86%) | 43/75 (46–68%) |

- Partir da pose inicial do workshop, e não da gravada no episódio, custa ~20 pontos a todas as políticas.
- Trocar a pose gravada pela pose mediana dos operadores muda pouco: de −8 a +6 pontos.
- O pi0.5 fica à frente do ACT nas quatro condições, de +8 a +29 pontos num run. Ele varia ±5 pontos entre runs.
- O dataset foi gravado na variante DR: a `eval`, sem domo e com o robô sempre laranja, está fora da distribuição de treino de todas as políticas.

## 8. Validação do ambiente, sem política

| Teste | Resultado |
|---|---|
| Replay cinemático dos 75 episódios (`scripts/replay_dataset.py`) | as juntas ficam nas faixas do MJCF; a ponta da garra no agarre e na soltura confere o alinhamento e a escala com as poses nominais |
| Replay dinâmico dos 75 episódios (`scripts/replay_dynamic.py`) | o PD do Isaac reproduz as juntas gravadas, p95 ≤ 1,5° |
| Replay das ações gravadas nas cenas reconstruídas (`--replay`) | 64/75 completam a tarefa; 61 dos 64 com o frasco no furo. As 75 demonstrações completam no Isaac |
| Demo roteirizada (`vials_lite.scripted`, 30 sementes) | 30/30 |
| Repouso (10 s, frascos no rack) | deslocamento máximo 0,2 mm, giro 0,6° |
| Mesma física em outra máquina (CPUs Intel × AMD, GPUs diferentes) | replay das ações do Isaac idêntico bit a bit, 100/100 episódios |
| Gap visual offline (`scripts/offline_gap.py`; mesmo instante, imagem do Isaac × do Vials-Lite) | distância mediana entre a garra prevista e a da demonstração na soltura: ACT 4,7 mm com as imagens do Isaac e 16,6 mm com as daqui; pi0.5 4,6 e 12,8 mm |

## 9. Conclusão

Para a ordem entre políticas e para o pi0.5 e o GR00T, o Vials-Lite substitui o Isaac dentro da resolução de 100 episódios, inclusive em cenas fora da amostra de calibração. Para as políticas ACT treinadas com imagens do Isaac, ele subestima em ~12 pontos fora da amostra, e essa parte acompanha as imagens de treino. Use o Vials-Lite para comparar políticas e diagnosticar falhas; para um número absoluto de uma política ACT treinada no Isaac, confirme no Isaac.

## Como reproduzir

- Matriz com sorteio, no Vials-Lite: `python -m evalkit.runner --policy <política> --variant {eval,dr-eval} --episodes 100 --seed 1000` ([manual.md](manual.md)).
- Colunas do Isaac, cenas gravadas, replays e comparação pareada: [validacao-isaac.md](validacao-isaac.md).
- Cenas do dataset: `scripts/fit_external_camera.py` → `scripts/reconstruct_scenes.py` → `evalkit.runner --scenes` ([scripts/README.md](../scripts/README.md)).
- Para registrar um resultado novo, siga o protocolo da seção 1 e anote na célula o que fugir dele. Guarde com o run: o `run.json` (versões, GPU e commit), o checkpoint exato e o comando.
