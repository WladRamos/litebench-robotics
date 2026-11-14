# Especificação da tarefa Vials-to-Rack

A cena vem da leitura do código e dos assets do [workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop) (Apache-2.0, commit `ce807d9`). Os valores estão em `src/vials_lite/spec.py`, a fonte única dos números da cena; este documento explica de onde cada um vem. Os arquivos citados entre crases sem caminho (`so101_env_cfg.py`, `mdp/terms.py` etc.) são do workshop, em `source/sim_to_real_so101/`.

A tarefa: o braço SO-101 pega um de três frascos deitados num tapete e o coloca em pé num rack de quatro furos, em até 7,5 s.

## Tempo

| Item | Valor | Origem |
|---|---|---|
| Passo da física | 1/120 s | `so101_env_cfg.py`: `sim.dt` |
| Decimação | 2 → **controle a 60 Hz** | `decimation = 2` |
| Episódio de avaliação | **450 steps** (7,5 s simulados) | `VialsToRackEvalEnvCfg`: `episode_length_s = 450/60` |
| Aquecimento da avaliação | 10 primeiros steps aplicam `initial_action` | `scripts/lerobot_eval.py` |
| fps declarado no dataset | 30 | `meta/info.json` |

**1 frame do dataset = 1 step de 60 Hz.** O gravador do workshop grava um frame por `env.step` (`lerobot_agent.py` chama `push_frame_to_buffer` a cada step), sem subamostrar, mas o dataset é rotulado 30 fps. Na avaliação, o cliente GR00T aplica uma ação do chunk por step de 60 Hz (`get_action` faz `popleft` a cada chamada). Treino e avaliação são coerentes entre si, e o Vials-Lite mantém essa semântica (`spec.FRAMES_PER_CONTROL_STEP`). O dataset não guarda timestamps que confirmem isso de outra forma; a conclusão vem do código.

## Robô

- Base em (−0,05; 0; 0), yaw 90°. Juntas do USD: `Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw` ↔ LeRobot: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.
- Pose inicial / `initial_action` (rad): (−0,2736; −0,6109; −0,0745; 1,5148; −1,6034; −0,1465).
- Atuadores implícitos (PD), `effort_limit_sim = 30`, kp/kd:

  | Junta | kp | kd |
  |---|---|---|
  | Rotation | 55 | 0,7 |
  | Pitch | 30 | 0,8 |
  | Elbow | 25 | 0,7 |
  | Wrist_Pitch | 12 | 0,5 |
  | Wrist_Roll | 7 | 0,5 |
  | Jaw | 4 | 0,3 |

  O MJCF do Menagerie usa kp 998,22, kv 2,731 e forcerange ±2,94 (os servos STS3215 do robô real). O Vials-Lite usa o PD do Isaac por padrão (`actuators="isaac"`): o dataset foi teleoperado no Isaac, e só esse preset reproduz as trajetórias gravadas (`scripts/replay_dynamic.py`). O preset `"menagerie"` fica para quem quiser o servo do robô real.

## Conversão de juntas (dataset ↔ simulador)

Porta de `utils/lerobot_interface.py`, em `src/vials_lite/joint_mapping.py`:

1. juntas 1–5: LeRobot [−100, 100] → [0, 1]; garra: [0, 100] → [0, 1];
2. → graus pelas faixas: pan [−110, 110], lift [−100, 100], elbow [−100, 90], wrist_flex [−95, 95], wrist_roll [−160, 160], gripper [−10, 100];
3. → radianos.

É um mapeamento afim que imita uma calibração.

**Alinhamento com o MJCF do Menagerie** (`scripts/check_alignment.py`): a cinemática é a mesma, com erro < 0,002 mm em todos os corpos. Mudam duas coisas:
- o frame da base: o MJCF vai em `spec.MJCF_BASE_POS` com yaw 0, e não na pose do USD com yaw 90°;
- o zero do wrist_roll: `JOINT_OFFSETS_MJCF_RAD`, q_mjcf = q_usd + 0,04868 rad; as outras juntas têm offset zero.

O replay cinemático dos 75 episódios do dataset (`scripts/replay_dataset.py`) confere isso e não sai das faixas do MJCF. Os limites são os do USD (Rotation ±110°, Pitch ±100°, Elbow −100/+90°, Wrist_Pitch ±95°, Wrist_Roll ±160°, Jaw −10/+100°), com duas exceções no MJCF: Elbow ±96,8° e Wrist_Roll deslocado pelo offset.

## Objetos

| Objeto | Pose padrão (m) | Rotação | Massa |
|---|---|---|---|
| vial_1 | (0,23; −0,08; 0,05) | euler (0, 90, 0)°: deitado | 0,02 kg, `angular_damping = 100` |
| vial_2 | (0,23; 0; 0,05) | idem | idem |
| vial_3 | (0,23; −0,16; 0,05) | idem | idem |
| rack_left | (0,18; 0,08; 0,06) | — | 0,2 kg declarados; o Isaac simula 0,886 kg (abaixo) |
| tapete | (0,22; 0; 0,032) | yaw 90° | — |
| lightstudio | (−0,1; 0; 0,0257) | — | — |

### Geometria dos assets

Extraída dos pontos das malhas USD por `scripts/inspect_workshop_assets.py` (usd-core); frames locais dos assets, metros.

**Frasco** (`Vial_opaque.usda`): tubo cônico de centrífuga de 50 mL com tampa de rosca, como pede a lista de materiais do curso. Origem na base do corpo cilíndrico, eixo +z ao longo do tubo.

| Parte | Raio | z local |
|---|---|---|
| ponta cônica | 3,6 mm → 15,6 mm | −0,017 → 0 |
| corpo | 15,6 mm (Ø 31 mm) | 0 → 0,085 |
| tampa | 17,0 mm (Ø 34 mm) | 0,085 → 0,0997 |

Comprimento total ≈ 0,117 m. No Isaac, o colisor é SDF da malha. A massa autorada é 0, e vale a do cfg (0,02 kg). O USD limita a velocidade linear do frasco a 2 m/s (`physxRigidBody:maxLinearVelocity`). O `extent` autorado no USD está desatualizado (~0,7 mm) e não deve ser usado.

**Rack** (`Vial_rack_simple.usda`): corpo x, y ∈ [0; 0,12], z ∈ [−0,006; 0,073]; colisor SDF. O cfg declara 0,2 kg, mas o Isaac Lab só aplica o `MassPropertiesCfg` a prims com `MassAPI`, que este USD não tem; o PhysX usa então a densidade padrão, 1000 kg/m³, sobre o volume da malha (886 cm³): 0,886 kg. Quatro furos em grade 2×2 (passo ~6 cm), entrada em z = 0,1:

| Slot | x | y |
|---|---|---|
| top_01 | 0,0298 | 0,0299 |
| top_02 | 0,0902 | 0,0299 |
| top_03 | 0,0902 | 0,0901 |
| top_04 | 0,0308 | 0,0901 |

**Tapete** (`mat.usda`): 0,457 × 0,305 × 0,006 m, preto.

### Como a cena vira MuJoCo

O MuJoCo transforma uma malha de colisão no seu fecho convexo, e os furos do rack sumiriam. Por isso a colisão é montada com primitivas, e as malhas do workshop ficam só para a aparência (`src/vials_lite/scene.py`):

- **Frasco:** dois cilindros (corpo Ø 31,2 mm, tampa Ø 34 mm) e a ponta em cone convexo (`VIAL_TIP_SHAPE`), 20 g. O `angular_damping = 100` do workshop vira um amortecimento D = c·I nos graus de liberdade rotacionais, com c = 100. Depois de cada passo, a velocidade linear é limitada a 2 m/s, como o `maxLinearVelocity` do PhysX.
- **Rack:** placa de fundo e, por furo, um anel de 16 caixas: assento cônico (raio 5 mm → 15 mm, onde a ponta do frasco encaixa), parede reta (raio 17 mm) e chanfro (até raio 23 mm no topo), mais o tampo plano e os cantos, medidos na malha do USD; 0,886 kg, a massa que o Isaac simula (`scene.RACK_MASS_MODE`; `"declared"` usa os 0,2 kg do cfg).
- **Contato:** atrito 0,5 em todos os geoms, o material padrão do Isaac Lab, que vale para todo corpo sem material próprio (caso de todos os USD da cena). A garra do Menagerie passa a atrito 0,5 com torção 0,005 e sem rolamento (`condim` 4). Sem a torção, o frasco pendula na garra em torno do eixo de pinça; no Isaac ele fica preso, porque o contato SDF do PhysX cobre uma área e as primitivas tocam em poucos pontos.
- **Tapete e chão:** o tapete é uma caixa estática; o chão é um plano invisível em z = 0,029 (fundo do tapete).
- **Aparência:** as malhas visuais do rack e do frasco (`assets/vials_lite/`) não têm massa nem contato. As cores dos USD são convertidas de linear para sRGB, como o Isaac mostra. O rack leva oclusão de ambiente gravada na malha (`scripts/bake_ambient_occlusion.py`): sem ela, os furos ficam claros, e no Isaac são escuros.

## Randomização no reset (`mdp/resets.py`)

Amostras uniformes somadas à pose padrão; rotação = `quat_padrão ⊗ quat(roll, pitch, yaw)`.

| Alvo | Faixas |
|---|---|
| rack | x ±0,04 · y ±0,01 · yaw ±0,5 rad |
| frascos (os não pré-colocados) | x ±0,04 · y ±0,01 · roll ±0,3 rad · z fixo em 0,05 |
| pré-colocação | com probabilidade 0,33, um frasco sorteado começa num furo sorteado (não conta como sucesso, porque nunca foi agarrado) |
| luz da lightbox | exposição U(−3, 1) |
| tapete | yaw ±0,1 rad (±0,3 na variante DR) |
| câmera de punho | distância focal U(12, 15) mm |
| câmera externa (`camera_mount`) | posição x, y ±0,02, z ±0,01 · rotação ±0,05 rad em cada eixo |
| só na DR | domo: exposição U(−4, 3), temperatura U(2500, 9500) K, HDRI sorteada (24, Poly Haven, CC0); cor do robô sorteada |

O Vials-Lite tem as duas variantes de avaliação do workshop: `eval` (`VialsToRackEvalEnvCfg`) e `dr-eval` (`VialsToRackEvalDREnvCfg`). A `dr-eval` tem o mesmo sorteio físico da `eval` e acrescenta a cor do robô, a luz do domo e mais giro do tapete. O domo daqui tem cor uniforme: não há equivalente às HDRIs.

### Lightbox e luz

- A lightbox é uma caixa de papel aberta em −x (`lightbox-simple.usd`), com as faces internas em x ∈ [−0,002; 0,502], y ∈ [−0,379; 0,381], z ∈ [0,002; 0,505] no frame do LightStudio (`spec.LIGHTBOX_*`). Fora dela, o fundo é preto. Uma `RectLight` de 0,30 × 0,01 m no topo é a única luz da `eval`; o domo só existe na DR.
- O MuJoCo não reproduz as intensidades do Isaac, que usa path tracing e tone mapping. As constantes equivalentes (`spec.LIGHTING`) foram calibradas contra a cor do piso e da parede do fundo nos 75 vídeos do dataset (`scripts/fit_lighting.py`), mantendo as faixas de sorteio do workshop. A exposição entra como 2^(0,2115 · e).
- Como o dataset inteiro foi gravado na DR, a calibração precisa de uma premissa para a `eval`: com exposição 0 e sem domo, o piso tem o brilho mediano do vídeo.

## Câmeras

- 640×480, pinhole, RGB (o Isaac também gera profundidade e segmentação; aqui não).
- **`ego`** (punho): em `Robot/gripper/gripper_cam`, offset (−0,005; 0,06; −0,062), rotação euler (−45, 0, 0)°, convenção OpenGL. Focal 13,5 mm, sorteada entre 12 e 15 mm no reset. O workshop não define a abertura do sensor; o Vials-Lite supõe a abertura horizontal padrão do `PinholeCameraCfg` do Isaac Lab, 20,955 mm. A câmera é montada nessa pose no corpo da garra, e não se usa a `wrist_cam` do Menagerie.
- **`external_D455`** (externa): o `task_env_cfg.py` usa a câmera como definida no USD (`spawn=None`). Ela vem de um payload remoto dos assets do Isaac 5.1 (`rsd455.usd`; [assets/README.md](../assets/README.md)). Composta com LightStudio ∘ LightBox ∘ camera_mount ∘ RSD455, a pose no mundo é (0,1376; 0,3297; 0,4886), com focal 1,93 mm e FOV horizontal ≈ 90,5°.
  - **Esses valores não reproduzem o vídeo do dataset.** Calibrados contra os 75 episódios (`scripts/fit_external_camera.py`, pela silhueta do robô): FOV vertical 51,5° e posição (0,0903; 0,3228; 0,4771), ≈ 4,7 cm a menos em x, com a orientação do USD. São os valores do `spec`.
  - A variação entre episódios é compatível com a randomização do `camera_mount`.
  - A origem da diferença no Isaac não foi identificada.

## Critério de sucesso (`mdp/terms.py`)

Porta em `src/vials_lite/success.py`, NumPy puro, com testes em `tests/test_success.py`.

1. **Agarre:** força de contato entre a mandíbula móvel (`Robot/jaw`) e o frasco > 2 N.
2. **Colocação:** um frasco conta como colocado quando, ao mesmo tempo:
   - o episódio já saiu do aquecimento (step ≥ 30);
   - |eixo z do frasco · z do mundo| > 0,7 (inclinação até ~45°);
   - no frame do rack, x ∈ [0; 0,12], y ∈ [0; 0,12] e z < 0,1;
   - o frasco foi agarrado na janela recente e não está agarrado agora.

   A marca de "colocado" é pegajosa.
3. **Confirmação:** a partir do primeiro "colocado", conta-se quantos steps seguidos algum frasco marcado continua vertical, dentro da caixa e solto. Com 25 steps seguidos, o episódio termina com sucesso (`terminated`). Se a condição falhar no meio, o contador zera.
4. `truncated` em 450 steps. A taxa de sucesso conta os episódios `terminated and not truncated`.

**O critério não verifica o furo.** Em pé sobre o tampo, com a ponta na borda de um furo, a origem do frasco fica em ~0,09 m no frame do rack, abaixo dos 0,1, e o episódio conta como sucesso. `scripts/success_seating.py` separa os sucessos em encaixado, no furo, sobre o tampo e só o frasco pré-colocado.

**A janela de "agarrado recentemente" é de ~10 steps, não 20.** No workshop, `vial_placed_on_rack` é chamada pelo termo de terminação e pelo termo de observação `subtask_terms.vial_placed`. O estado da função (um buffer circular de 20 posições) é compartilhado e avança duas vezes por step. `SuccessTracker(history_writes_per_step=2)` reproduz isso, e o teste `test_effective_grasp_window_depends_on_double_call_quirk` mostra um caso em que a diferença muda o resultado. A conclusão vem da leitura do código; não foi conferida instrumentando o Isaac.

## Protocolo de avaliação do workshop (`scripts/lerobot_eval.py`)

- Semente padrão 1984 (Python, NumPy, torch e ambiente).
- `--action_horizon 16` por padrão: o cliente executa as 16 primeiras ações de cada chunk.
- Instrução: *"Pick up the vial and place it in the yellow rack"*, a da página "Sim Evaluation" do curso para as variantes `Eval` e `DR-Eval`. O padrão do script diz "...in the rack", que é o texto do dataset. O Vials-Lite usa a do curso por padrão (`--instruction` muda).
- `--rename_map '{"external_D455": "front", "ego": "wrist"}'` para o GR00T do curso.

## Dataset do curso

`sreetz-nv/so101_teleop_vials_rack_left`: LeRobot v3.0, `robot_type = so101_follower`, 75 episódios e 18.250 frames, fps declarado 30. `observation.state` e `action` são vetores de 6 floats com os nomes `*.pos` do LeRobot. Os vídeos `observation.images.ego` e `observation.images.external_D455` são 640×480 em AV1.

O dataset não grava o estado dos objetos nem recompensa: as poses de frascos e rack de cada episódio precisam ser reconstruídas pelo vídeo (`scripts/reconstruct_scenes.py`). Foi gravado na variante DR, com o arranjo da lightbox ~5 cm mais longe do robô que no código do workshop (`spec.DATASET_LAYOUT_OFFSET`, opção `--layout dataset`). A licença não é declarada no card.
