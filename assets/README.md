# assets/

Versionado: `vials_lite/` com as malhas **só visuais** do rack e do frasco (OBJ, ~900 kB), derivadas
dos USD do workshop (Apache-2.0, atribuição no `NOTICE`) por `scripts/export_workshop_meshes.py`
(precisa do extra `usd` e do clone do workshop abaixo), seguido de `scripts/bake_ambient_occlusion.py`
(oclusão de ambiente do rack, gravada no OBJ). O resto é baixado localmente para
`assets/third_party/` (ignorado pelo git):

| O quê | De onde | Licença | Como |
|---|---|---|---|
| SO-101 (MJCF + malhas) | [MuJoCo Menagerie `robotstudio_so101`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101) | Apache-2.0 | `git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git assets/third_party/mujoco_menagerie` e `git -C assets/third_party/mujoco_menagerie sparse-checkout set robotstudio_so101` (só a pasta do SO-101; o clone raso completo também funciona) |
| Frasco, rack, tapete, lightbox (USD) | [Workshop NVIDIA](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop) `source/sim_to_real_so101/assets/usd` | Apache-2.0 (repositório) | clonar **ao lado** deste repositório (`../Sim-to-Real-SO-101-Workshop`), `git checkout ce807d9` e `git lfs pull --include="source/sim_to_real_so101/assets/usd/*"` (os USD são LFS, ~57 MB). Sem `git-lfs`: baixar cada ponteiro de `https://media.githubusercontent.com/media/isaac-sim/Sim-to-Real-SO-101-Workshop/ce807d9/<caminho>` e conferir o SHA-256 contra o `oid` do ponteiro |
| HDRIs | idem, `assets/hdri` | CC0 (Poly Haven) | idem (ainda não usados) |
| Câmera RealSense D455 (`rsd455.usd`, payload da câmera externa) | assets do Isaac Sim 5.1: `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Sensors/Intel/RealSense/rsd455.usd` (URL lida do `lightbox-simple.usd`) | não declarada (termos dos assets do Isaac Sim) | baixar para `assets/third_party/isaac_sensors/rsd455.usd` (4,5 MB); só é lido localmente, para obter pose e intrínseca da câmera, e não é redistribuído |

Licença dos USD: não têm licença própria e o `THIRD_PARTY_NOTICES`
do workshop não os menciona, então valem os termos Apache-2.0 do repositório. A colisão do rack e do
frasco continua em primitivos (`scene.py`); o tapete continua uma caixa.
