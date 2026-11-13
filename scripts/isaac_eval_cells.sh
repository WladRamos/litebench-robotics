#!/usr/bin/env bash
# Células do Isaac para uma política do LeRobot: o lerobot_eval.py do workshop, sem mudanças, consultando
# a ponte no protocolo do GR00T (evalkit.groot_bridge).
#
# Uso, da raiz deste repositório, na máquina do Isaac (Linux):
#   ISAAC_PY=<python do ambiente do Isaac> WORKSHOP=<clone do workshop no ce807d9> \
#     scripts/isaac_eval_cells.sh <nome> <política> [episódios=100] [variantes="Eval DR-Eval"]
# Exemplo:
#   scripts/isaac_eval_cells.sh act_lite outputs/train/act_lite_66ep/checkpoints/050000/pretrained_model
#
# Para cada variante: sobe a ponte, roda o cliente, espera a linha "Success Rate" (o lerobot_eval.py não
# sai sozinho no fim) e encerra os dois. Saídas em outputs/isaac/<nome>/: <variante>.client.log (o número),
# <variante>.ponte.log e .ponte.jsonl (consultas por episódio: confere o --action_horizon) e meta.txt.
# Com RECORD=1 (padrão), o cliente roda pelo gravador (scripts/isaac_recorder.py), que não muda o
# arquivo do workshop e grava em <variante>.rec/ o estado por step e a cena de cada episódio; RECORD=0
# roda o cliente direto, sem gravar.
#
# GR00T do curso: com GROOT_DIR=<clone do Isaac-GR00T no ead5283, com .venv>, o servidor é o
# gr00t/eval/run_gr00t_server.py, e <política> é a pasta do checkpoint. O protocolo do curso pede
# HORIZON=16 (padrão: 100, o da ponte) e RENAME_MAP='{"external_D455": "front", "ego": "wrist"}'.
#
# Replay no Isaac: com REPLAY=1, <política> é a pasta de uma avaliação gravada
# (outputs/isaac/<nome>), e a ponte devolve as ações de <política>/<variante>.rec em vez de consultar uma
# política (python -m evalkit.groot_bridge --replay). Use a mesma SEED e o mesmo número de episódios da
# gravação. VIAL_SDF=N troca a resolução do SDF de colisão dos frascos (o workshop usa 20; exige o gravador).
# NOISE=σ soma às ações gravadas um ruído de σ graus, semeado por episódio (o piso; groot_bridge.action_noise).
#   REPLAY=1 SEED=2026 scripts/isaac_eval_cells.sh rp_pi05_s2026 outputs/isaac/pi05_s2026
#   REPLAY=1 SEED=2026 VIAL_SDF=64 scripts/isaac_eval_cells.sh rp64_pi05_s2026 outputs/isaac/pi05_s2026
#   REPLAY=1 SEED=2026 NOISE=0.01 scripts/isaac_eval_cells.sh rpn_pi05_s2026 outputs/isaac/pi05_s2026
#
# EXTERNAL=1: não sobe servidor; usa o que já estiver ouvindo em PORT (<política> só vai para o meta.txt). Serve
# para um servidor GR00T atender dois clientes em paralelo (cada um numa variante) ou para uma ponte noutra
# máquina, por túnel SSH até 127.0.0.1:PORT.
set -euo pipefail

name=${1:?nome da política (pasta de saída)}
policy=${2:?pasta ou repo do Hub da política}
episodes=${3:-100}
variants=${4:-"Eval DR-Eval"}
port=${PORT:-5555}
horizon=${HORIZON:-100}
rename=${RENAME_MAP:-}
groot_dir=${GROOT_DIR:-}
record=${RECORD:-1}
seed=${SEED:-1984}  # o padrão do lerobot_eval.py
frames=${FRAMES:-0}  # com o gravador: vídeo das câmeras nos N primeiros episódios
replay=${REPLAY:-0}
vial_sdf=${VIAL_SDF:-}
noise=${NOISE:-0}
external=${EXTERNAL:-0}
instruction="Pick up the vial and place it in the yellow rack"  # o texto da tarefa na página do curso
: "${ISAAC_PY:?defina ISAAC_PY, o python do ambiente do Isaac}"
: "${WORKSHOP:?defina WORKSHOP, o clone do workshop}"
py=.venv/bin/python  # o do Vials-Lite (uv sync --locked --extra lerobot --extra bridge)
if [ -n "$vial_sdf" ] && [ "$record" != 1 ]; then echo "VIAL_SDF exige RECORD=1"; exit 1; fi

out=outputs/isaac/$name
mkdir -p "$out"
{
    echo "data: $(date -Iseconds)"
    echo "política: $policy"
    echo "episódios: $episodes; variantes: $variants; semente: $seed; --action_horizon $horizon; texto: $instruction"
    echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
    echo "vials-lite: $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (com mudanças)')"
    echo "workshop: $(git -C "$WORKSHOP" rev-parse --short HEAD)"
    echo "gravador: RECORD=$record; vídeo: FRAMES=$frames"
    [ "$replay" = 1 ] && echo "replay: ações de $policy/<variante>.rec; ruído $noise°"
    [ -n "$vial_sdf" ] && echo "frascos: sdfResolution $vial_sdf"
    [ "$external" = 1 ] && echo "servidor: externo, já no ar na porta $port"
    [ -n "$groot_dir" ] && echo "servidor: Isaac-GR00T $(git -C "$groot_dir" rev-parse --short HEAD)"
    [ -n "$rename" ] && echo "rename_map: $rename"
} >> "$out/meta.txt"

server=""
client=""
cleanup() {
    [ -n "$client" ] && kill "$client" 2>/dev/null || true
    [ -n "$server" ] && kill "$server" 2>/dev/null || true
}
trap cleanup EXIT

for variant in $variants; do
    log=$out/$variant
    if [ "$external" = 1 ]; then
        echo "[$variant] servidor externo na porta $port; cliente rodando (acompanhe: tail -f $log.client.log)"
    elif [ -n "$groot_dir" ]; then
        "$groot_dir/.venv/bin/python" "$groot_dir/gr00t/eval/run_gr00t_server.py" --model-path "$policy" \
            --host 127.0.0.1 --port "$port" > "$log.ponte.log" 2>&1 &
        ready="Server is ready"
    elif [ "$replay" = 1 ]; then
        "$py" -m evalkit.groot_bridge --replay "$policy/$variant.rec" --action-steps "$horizon" \
            --action-noise-deg "$noise" \
            --port "$port" --log "$log.ponte.jsonl" > "$log.ponte.log" 2>&1 &
        ready="ouvindo"
    else
        "$py" -m evalkit.groot_bridge --policy "$policy" --port "$port" --log "$log.ponte.jsonl" \
            > "$log.ponte.log" 2>&1 &
        ready="ouvindo"
    fi
    if [ "$external" != 1 ]; then
        server=$!
        until grep -q "$ready" "$log.ponte.log"; do
            kill -0 "$server" 2>/dev/null || { cat "$log.ponte.log"; exit 1; }
            sleep 2
        done
        echo "[$variant] ponte pronta; cliente rodando (acompanhe: tail -f $log.client.log)"
    fi
    # sem buffer: o simulation_app.close() encerra o processo sem esvaziar o stdout, e a linha
    # "Success Rate" se perdia no arquivo
    eval_py=$WORKSHOP/source/sim_to_real_so101/scripts/lerobot_eval.py
    if [ "$record" = 1 ]; then
        launch=("$ISAAC_PY" scripts/isaac_recorder.py --out "$log.rec" --frames "$frames"
                ${vial_sdf:+--vial-sdf-resolution "$vial_sdf"} --script "$eval_py" --)
    else
        launch=("$ISAAC_PY" "$eval_py")
    fi
    PYTHONUNBUFFERED=1 "${launch[@]}" \
        --task "Lerobot-So101-Teleop-Vials-To-Rack-$variant" --num_envs 1 --headless --seed "$seed" \
        --num_episodes "$episodes" --action_horizon "$horizon" --lang_instruction "$instruction" \
        ${rename:+--rename_map "$rename"} --policy_port "$port" > "$log.client.log" 2>&1 &
    client=$!
    until grep -q "Success Rate:" "$log.client.log"; do
        kill -0 "$client" 2>/dev/null || break
        sleep 10
    done
    sleep 5
    kill "$client" 2>/dev/null || true
    wait "$client" 2>/dev/null || true
    client=""
    kill "$server" 2>/dev/null || true
    wait "$server" 2>/dev/null || true
    server=""
    result=$(grep "Success Rate:" "$log.client.log" | tail -1 || true)
    echo "[$variant] ${result:-SEM RESULTADO: veja $log.client.log}" | tee -a "$out/meta.txt"
done
