# Instalação

O projeto usa Python 3.12 e o [uv](https://docs.astral.sh/uv/), que cria o ambiente a partir do `uv.lock` com as versões exatas usadas nos resultados. Funciona em Linux e Windows. Os comandos abaixo são para bash; as diferenças no PowerShell aparecem quando existem.

## 1. uv e o repositório

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh           # Linux; no Windows: winget install astral-sh.uv
git clone https://github.com/WladRamos/litebench-robotics.git && cd litebench-robotics
```

O uv baixa o Python 3.12 sozinho, se ele não estiver instalado.

## 2. Extras

O pacote base só depende do NumPy. O resto vem por extras, combináveis num único `uv sync`:

| Extra | Traz | Para quê |
|---|---|---|
| `dev` | pytest, ruff | testes e lint |
| `sim` | MuJoCo, Gymnasium | a cena, o ambiente e o runner |
| `lerobot` | LeRobot 0.6.1 (com pi0.5 e treino), PyTorch | carregar e avaliar políticas, `lerobot-eval`, `lerobot-train` |
| `bridge` | pyzmq, msgpack | a ponte no protocolo do GR00T (validação no Isaac) |
| `usd` | usd-core | extrair dados e malhas dos USD do workshop |

```bash
uv sync --locked --extra dev                                   # só o núcleo: roda em qualquer máquina
uv sync --locked --extra dev --extra sim                       # cena e ambiente
uv sync --locked --extra dev --extra sim --extra lerobot       # avaliar políticas
```

Use sempre `--locked`: ele recusa instalar se o `uv.lock` não corresponder ao `pyproject.toml`. Um `uv sync` com outros extras remove os que não forem pedidos; para rodar um comando sem sincronizar de novo, use `uv run --no-sync`.

A versão do LeRobot é fixada em 0.6.1. Treine e avalie com a mesma versão: checkpoints de outras versões podem carregar sem erro e se comportar diferente.

## 3. Assets de terceiros

O robô vem do [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) e não é versionado aqui. Sem ele, os testes de cena são pulados e o ambiente não abre.

```bash
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git assets/third_party/mujoco_menagerie
```

Outro lugar também serve: aponte a variável `VIALS_LITE_MENAGERIE` para o clone.

As malhas visuais do rack e do frasco já vêm no repositório (`assets/vials_lite/`). Os USD do workshop só são necessários para regenerá-las ou extrair dados da cena de novo; as instruções estão em [assets/README.md](../assets/README.md).

## 4. GPU e render

O MuJoCo renderiza por OpenGL. A física roda na CPU; a GPU só desenha as duas câmeras de 640×480.

- **Linux:** use `MUJOCO_GL=egl`, que não precisa de janela nem de monitor. O runner do evalkit já escolhe `egl` se a variável não estiver definida.
- **Windows:** não defina `MUJOCO_GL`. O padrão (GLFW/WGL) funciona sem abrir janela.

**Confira qual GPU está renderizando:**

```bash
MUJOCO_GL=egl uv run python scripts/bench_render.py --menagerie assets/third_party/mujoco_menagerie
```

O script imprime o `GL_RENDERER`. Se aparecer Intel, AMD, Mesa ou llvmpipe em vez de NVIDIA, o render está na GPU integrada ou em software: um episódio leva ~85 s em vez de ~1,7 s, e as imagens não batem com as de outros runs. O runner recusa rodar nesse caso (a opção `--any-renderer` libera).

**Notebook com GPU híbrida (Intel + NVIDIA):**

- **Linux:** primeiro confira se o driver NVIDIA está carregado (`nvidia-smi`; depois de atualizar o kernel, reinicie). Com o driver carregado, o EGL costuma escolher a NVIDIA sozinho. Se não escolher, tente `MUJOCO_EGL_DEVICE_ID=<n>` ou `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`.
- **Windows:** em "Configurações de gráficos", marque como "Alto desempenho" o `python.exe` do `.venv` **e** o interpretador base do uv (`%APPDATA%\uv\python\cpython-3.12...\python.exe`): o do `.venv` é só um lançador, e quem abre o contexto OpenGL é o base. A docstring de `scripts/bench_render.py` tem os detalhes.

**PyTorch e CUDA** (extra `lerobot`):

- O PyTorch do `uv.lock` é compilado com CUDA 13 e precisa de driver NVIDIA ≥ 580.
- As RTX 50xx (Blackwell) exigem PyTorch com CUDA ≥ 12.8.
- Para conferir:

  ```bash
  uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(), torch.cuda.get_arch_list())"
  ```

  Na RTX 50xx, a lista precisa conter `sm_120`.
- Com driver mais antigo (por exemplo 570, CUDA 12.8), troque só a compilação do PyTorch no `.venv`, sem mexer no `uv.lock`:

  ```bash
  uv pip install --python .venv/bin/python --index https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match "torch==2.11.0+cu128" "torchvision==0.26.0+cu128"
  ```

## 5. Políticas e dataset do curso

As políticas públicas são baixadas do Hugging Face Hub na primeira vez que o runner as usa:

| Política | Repositório no Hub | Observação |
|---|---|---|
| ACT do curso | `arjunjuneja/act_vials_50k` | público, ~200 MB |
| pi0.5 | `ltchang/pi05_so101_vials_canonical75_cont_15k` | o repositório guarda os checkpoints intermediários (~150 GB); baixe só a raiz, abaixo |

O pi0.5 usa o tokenizer do `google/paligemma-3b-pt-224`, de acesso restrito: aceite a licença na página do modelo e faça `hf auth login` (o `hf` vem com o extra `lerobot`). Para baixar só o checkpoint final:

```bash
uv run hf download ltchang/pi05_so101_vials_canonical75_cont_15k --exclude "checkpoints/*" \
    --local-dir checkpoints/pi05_vials
```

O dataset do curso só é necessário para os scripts de replay, reconstrução de cenas e gap visual:

```bash
uv run hf download sreetz-nv/so101_teleop_vials_rack_left --repo-type dataset \
    --local-dir datasets/so101_teleop_vials_rack_left
```

São 75 episódios: parquets de juntas e ações (~1,6 MB) e vídeos AV1 das duas câmeras. O dataset não declara licença; use-o pelo Hub e não o redistribua. `datasets/`, `checkpoints/` e `outputs/` são ignorados pelo git.

## 6. Verificar a instalação

```bash
uv run pytest -q          # todos verdes; pulados só os que dependem de algo não instalado
uv run ruff check .
```

Os testes dizem por que foram pulados (`-rs`): sem MuJoCo, sem o Menagerie, sem o LeRobot, sem o pyzmq ou sem o usd-core. O núcleo (spec, juntas, sucesso, estatística, fases, falhas) roda sem nenhum deles; é o que o CI executa em Linux e Windows.

## Problemas comuns

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| `MJCF do Menagerie não encontrado` | o Menagerie não foi clonado | seção 3, ou `VIALS_LITE_MENAGERIE` |
| o runner recusa rodar: render fora da NVIDIA | GPU híbrida, ou driver não carregado | seção 4 |
| `torch.cuda.is_available()` dá `False` | driver antigo para o PyTorch do `uv.lock` | seção 4, troca para cu128 |
| `no kernel image is available` numa RTX 50xx | PyTorch sem `sm_120` | PyTorch com CUDA ≥ 12.8 |
| erro 401/403 ao carregar o pi0.5 | tokenizer do PaliGemma restrito | aceitar a licença e `hf auth login` |
| `ModuleNotFoundError: pyarrow` ou `av` num script | dependências só dos scripts | rodar com `uv run --with pyarrow --with av ...`, como na docstring do script |
| o `uv sync` removeu o LeRobot | o sync instala só os extras pedidos | repetir os extras, ou `uv run --no-sync` |
