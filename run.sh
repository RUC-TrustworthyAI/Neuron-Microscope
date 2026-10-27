#!/usr/bin/env bash
set -euo pipefail
FRAMEWORK_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export NEURON_MICROSCOPE_ROOT=${NEURON_MICROSCOPE_ROOT:-$FRAMEWORK_DIR}
TASK_PYTHON=${NEURON_PAPER_PYTHON:-python3}
if [[ -n "${NEURON_PAPER_PYTHONPATH:-}" ]]; then
    export PYTHONPATH="$NEURON_PAPER_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
fi
STAGES=(01-interpretation:interpret 02-experiments:experiment 03-revision:revise)

files() { "$TASK_PYTHON" "$FRAMEWORK_DIR/tools/launch_files.py" "$@"; }

prepare_batch() {
    local batch=$1 unit spec stage skill
    shift
    files make-batch "$batch" "${STAGES[*]}" "$@"
    if [[ -n "${initial_handoff:-}" ]]; then
        files set-handoff "$batch" "$1" "$initial_handoff"
    fi
    for unit in "$@"; do
        for spec in "${STAGES[@]}"; do
            stage=${spec%%:*}; skill=${spec#*:}
            files make-stage "$batch" "$unit" "$stage" "$skill"
        done
    done
    printf '%s\n' "$batch"
}

wait_for() {
    local path=$1 pid=$2 i
    for ((i=0;i<150;i++)); do
        [[ -e "$path" ]] && return 0
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 0.2
    done
    return 1
}

# Build the actual docker command here; Python tools do not launch containers.
build_container_command() {
    local folder=$1 unit=$2 stage=$3 skill=$4 previous=$5 socket_dir=$6
    local existing_only
    existing_only=$(files field "$folder/host-session.json" existing_only)
    CONTAINER="np-${unit,,}-${stage%%-*}-$(cat /proc/sys/kernel/random/uuid)"
    CMD=(docker run --name "$CONTAINER" --init --network "$NETWORK"
        --add-host "agent-host:$GATEWAY" --cpus 4 --memory 8g
        --cap-drop ALL --security-opt no-new-privileges
        -e DEEPSEEK_API_KEY -e PATH=/control/bin:/usr/local/bin:/usr/bin:/bin
        -e "DEEPSEEK_BASE_URL=http://agent-host:$METER_PORT/$unit/$stage/v1"
        --mount "type=bind,src=$folder/control,dst=/control,readonly"
        --mount "type=bind,src=$folder/control/skills/$skill,dst=/skills/$skill,readonly"
        --mount "type=bind,src=$socket_dir,dst=/run/neuron-paper,readonly")
    if [[ "$stage" == 01-interpretation ]]; then
        CMD+=(--mount "type=bind,src=$folder/input/report,dst=/inputs/report,readonly")
    elif [[ "$existing_only" != true ]]; then
        CMD+=(--mount "type=bind,src=$(files field "$folder/host-session.json" activations),dst=/inputs/activations,readonly")
    fi
    if [[ -n "$previous" ]]; then
        CMD+=(--mount "type=bind,src=$previous,dst=/inputs/handoff,readonly")
    fi
    if [[ "$stage" == 02-experiments && "$existing_only" != true ]]; then
        CMD+=(--mount "type=bind,src=$(files field "$folder/host-session.json" model),dst=/inputs/model,readonly")
    fi
    CMD+=(--mount "type=bind,src=$folder/workspace,dst=/workspace"
          --mount "type=bind,src=$folder/logs/agent,dst=/logs/agent"
          -w /workspace --entrypoint /usr/local/bin/python3 "$IMAGE" /control/agent.py)
}

launch_stage() (
    local folder=$1 unit=$2 stage=$3 skill=$4 previous=$5
    local socket_dir service_pid='' code=0 CONTAINER='' CMD=()
    local state_dir
    state_dir=$(files field "$folder/host-session.json" state_dir)
    mkdir -p "$state_dir/np-sockets"
    socket_dir=$(mktemp -d "$state_dir/np-sockets/XXXXXXXXXXXX")
    cleanup_stage() {
        if [[ -n "$CONTAINER" ]]; then
            docker stop --time 10 "$CONTAINER" >/dev/null 2>&1 || true
            docker rm "$CONTAINER" >/dev/null 2>&1 || true
        fi
        if [[ -n "$service_pid" ]]; then
            kill "$service_pid" 2>/dev/null || true
            wait "$service_pid" 2>/dev/null || true
        fi
        rm -f -- "$socket_dir/tool.sock"
        rmdir -- "$socket_dir" 2>/dev/null || true
    }
    trap cleanup_stage EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    if [[ "$stage" == 01-interpretation ]]; then
        files stage-input "$folder" || return $?
    fi
    files attach "$folder" "$previous" || return $?
    files record "$folder/status.json" running "stage=$stage" || return $?
    "$TASK_PYTHON" "$FRAMEWORK_DIR/tools/services.py" session \
        "$folder/host-session.json" "$socket_dir/tool.sock" "$AGENT_WALL" \
        >"$folder/logs/tool-service.log" 2>&1 &
    service_pid=$!
    wait_for "$socket_dir/tool.sock" "$service_pid" || return 1
    build_container_command "$folder" "$unit" "$stage" "$skill" "$previous" "$socket_dir"
    files launch "$folder" "$CONTAINER" "${CMD[@]}" || return $?
    timeout --foreground --signal=TERM --kill-after=30 "$AGENT_WALL" \
        "${CMD[@]}" >"$folder/logs/container.log" 2>&1 || code=$?
    files record "$folder/CONTROL_TERMINAL.json" exited "exit_code=$code" || return $?
    return "$code"
)

# One neuron: finish and collect each stage before starting the next container.
execute_unit() {
    local batch=$1 unit=$2 spec stage skill folder previous='' code
    previous=$(files initial-handoff "$batch" "$unit")
    for spec in "${STAGES[@]}"; do
        stage=${spec%%:*}; skill=${spec#*:}; folder="$batch/units/$unit/$stage"
        if [[ -f "$folder/handoff/handoff.json" && -f "$folder/status.json" ]] &&
            [[ "$(files field "$folder/status.json" status)" == complete ]]; then
            previous="$folder/handoff"
            continue
        fi
        if launch_stage "$folder" "$unit" "$stage" "$skill" "$previous"; then
            if files handoff "$folder" >"$folder/handoff.log" 2>&1; then
                previous="$folder/handoff"
            else
                code=$?
                files record "$batch/units/$unit/PIPELINE_TERMINAL.json" failed \
                    "stage=$stage" "exit_code=$code" "error=Handoff could not be assembled"
                return "$code"
            fi
        else
            code=$?
            files record "$batch/units/$unit/PIPELINE_TERMINAL.json" failed \
                "stage=$stage" "exit_code=$code"
            return "$code"
        fi
    done
    if [[ "$previous" == */03-revision/handoff ]]; then
        if files finalize-costs "$batch" "$unit" "$previous"; then
            :
        else
            code=$?
            files record "$batch/units/$unit/PIPELINE_TERMINAL.json" failed \
                "stage=publication" "exit_code=$code"
            return "$code"
        fi
    fi
    files finish-unit "$batch/units/$unit" "$previous"
}

execute_batch() (
    local batch=$1 scratch meter_pid='' unit pid code=0
    local NETWORK='' GATEWAY='' METER_PORT='' IMAGE='' AGENT_WALL='' START_NEXT_AFTER='' pids=()
    local state_dir
    state_dir=$(files state-dir "$batch")
    mkdir -p "$state_dir/np-run"
    scratch=$(mktemp -d "$state_dir/np-run/XXXXXXXXXXXX")
    cleanup_batch() {
        for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
        for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
        if [[ -n "$meter_pid" ]]; then
            kill "$meter_pid" 2>/dev/null || true; wait "$meter_pid" 2>/dev/null || true
        fi
        if [[ -n "$NETWORK" ]]; then docker network rm "$NETWORK" >/dev/null 2>&1 || true; fi
        rm -rf -- "$scratch"
    }
    trap cleanup_batch EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    mapfile -t STAGES < <(files stages "$batch/manifest.json")
    START_NEXT_AFTER=$(files start-next-after "$batch")
    IMAGE=$(files field "$batch/manifest.json" image)
    AGENT_WALL=$(files field "$batch/manifest.json" agent_wall_seconds)
    "$TASK_PYTHON" "$FRAMEWORK_DIR/tools/services.py" meter "$batch" "$scratch/meter.json" \
        >"$batch/logs/meter.log" 2>&1 &
    meter_pid=$!
    wait_for "$scratch/meter.json" "$meter_pid"
    METER_PORT=$(files field "$scratch/meter.json" port)
    export NEURON_PAPER_METER_PORT="$METER_PORT"
    DEEPSEEK_API_KEY=$(files field "$scratch/meter.json" token); export DEEPSEEK_API_KEY
    NETWORK="neuron-paper-$(cat /proc/sys/kernel/random/uuid)"
    docker network create --internal "$NETWORK" >/dev/null
    GATEWAY=$(docker network inspect --format '{{(index .IPAM.Config 0).Gateway}}' "$NETWORK")
    files record "$batch/launch.json" running "pid=$BASHPID" "network=$NETWORK"
    while IFS= read -r unit; do
        execute_unit "$batch" "$unit" &
        pid=$!;pids+=("$pid")
        if [[ "$START_NEXT_AFTER" == 02-experiments ]]; then
            # Host-side scheduling only. No next agent exists while this gate waits.
            while kill -0 "$pid" 2>/dev/null &&
                  [[ ! -f "$batch/units/$unit/02-experiments/handoff/handoff.json" ]]; do
                sleep 1
            done
        fi
    done <"$batch/units.txt"
    for pid in "${pids[@]}"; do wait "$pid" || code=1; done
    pids=()
    files finish-batch "$batch"
    return "$code"
)

main() {
    local command=${1:---help} batch='' output='' through=revise from=interpret initial_handoff='' reuse_measurements='' first=0 last=2 units=()
    [[ $# -gt 0 ]] && shift
    case "$command" in
        prepare)
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --settings) export NEURON_MICROSCOPE_SETTINGS=$(realpath -- "$2"); shift 2 ;;
                    --output) output=$2; shift 2 ;;
                    --through) through=$2; shift 2 ;;
                    --from) from=$2; shift 2 ;;
                    --handoff) initial_handoff=$(realpath -- "$2"); shift 2 ;;
                    --reuse-measurements) reuse_measurements=$(realpath -- "$2"); shift 2 ;;
                    --reuse-completed-experiments) reuse_measurements=$(realpath -- "$2"); export NEURON_PAPER_REUSE_COMPLETED=1; shift 2 ;;
                    --neurons) shift; units=("$@"); break ;;
                    *) echo "Unknown argument: $1" >&2; return 2 ;;
                esac
            done
            if [[ -z "$output" || ${#units[@]} -eq 0 ]]; then echo 'Specify --output and --neurons.' >&2; return 2; fi
            case "$from" in
                interpret) first=0 ;; experiment) first=1 ;; revise) first=2 ;;
                *) echo Use --from interpret, experiment or revise. >&2; return 2 ;;
            esac
            case "$through" in
                interpret) last=0 ;; experiment) last=1 ;; revise) last=2 ;;
                *) echo Use --through interpret, experiment or revise. >&2; return 2 ;;
            esac
            if (( first > last )); then echo --from must precede --through. >&2; return 2; fi
            if (( first > 0 )) && [[ -z "$initial_handoff" || ${#units[@]} -ne 1 ]]; then
                echo Continuing a stage requires --handoff DIR and one neuron. >&2; return 2
            fi
            STAGES=("${STAGES[@]:first:last-first+1}")
            output=$(realpath -m -- "$output")
            export NEURON_PAPER_EXISTING_MEASUREMENTS="$reuse_measurements"
            prepare_batch "$output" "${units[@]}"
            ;;
        execute)
            if [[ ${1:-} != --run || $# != 2 ]]; then echo 'Specify --run DIR.' >&2; return 2; fi
            batch=$(realpath -- "$2")
            unset NEURON_MICROSCOPE_SETTINGS
            # Execute the prepared snapshot, preserving the real project root.
            if [[ "$FRAMEWORK_DIR/run.sh" != "$batch/framework-source/run.sh" ]]; then
                exec "$batch/framework-source/run.sh" execute --run "$batch"
            fi
            execute_batch "$batch"
            ;;
        publish)
            local published_unit='' published_output=''
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --run) batch=$(realpath -- "$2"); shift 2 ;;
                    --neuron) published_unit=$2; shift 2 ;;
                    --output) published_output=$(realpath -m -- "$2"); shift 2 ;;
                    *) echo "Unknown argument: $1" >&2; return 2 ;;
                esac
            done
            if [[ -z "$batch" || -z "$published_unit" || -z "$published_output" ]]; then
                echo 'Specify --run DIR --neuron ID --output NEW_DIRECTORY.' >&2; return 2
            fi
            if [[ "$FRAMEWORK_DIR/run.sh" != "$batch/framework-source/run.sh" ]]; then
                exec "$batch/framework-source/run.sh" publish --run "$batch" --neuron "$published_unit" --output "$published_output"
            fi
            files publish "$batch" "$published_unit" "$published_output"
            ;;
        -h|--help)
            printf '%s\n' 'Usage: ./run.sh prepare --output DIR [--settings FILE] [--from interpret|experiment|revise] [--through interpret|experiment|revise] [--handoff DIR] [--reuse-measurements DIR] --neurons L55_N11703 ...' \
                '       ./run.sh execute --run DIR' \
                '       ./run.sh publish --run DIR --neuron ID --output NEW_DIRECTORY'
            ;;
        *) echo 'Use prepare or execute; see --help.' >&2; return 2 ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
