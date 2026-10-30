# Running

## Host and container

Use Linux and Bash 4+ for container orchestration. The offline demo can also run on macOS. Python must be 3.10 or newer.

The host reads saved arrays, serves evidence and runs GPU workers. Containers run the agent only; installing CUDA in the agent image is unnecessary. Each container receives a single skill and an isolated workspace. Inputs and handoffs are read-only mounts. The first agent receives only its own report directory; full activations and weights are not mounted there. Host preparation uses the archive and tokenizer before that agent starts.

Install host dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
# Choose a CUDA wheel compatible with your NVIDIA driver.
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements/gpu.txt
```

The recorded development environment used torch 2.7.1+cu118, Transformers 4.57.6, tokenizers 0.22.2, NumPy 2.2.6, PyArrow 21.0.0 and SciPy 1.15.3. The adapter shards complete dense-Qwen3 layers over the configured GPU group. Available memory depends on model size, input prefixes and backward storage; changing the group does not change the model identity.

Build the agent image:

```bash
docker build -t neuron-microscope:local .
```

The Dockerfile installs jq, ripgrep, Bash and DeepSeek Harness SDK/runtime 0.2.1rc2 from public packages. `--build-arg BASE_IMAGE=python:3.11-slim` can use an already cached compatible public Python base.

The default Debian package source uses HTTPS. On a network with slow access to upstream registries, `--build-arg DEBIAN_MIRROR=https://mirrors.tuna.tsinghua.edu.cn` and `--build-arg PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple` select alternative package mirrors for the build; runtime settings are unaffected.

## Configuration

Copy `settings.json` to `settings.local.json`. Relative paths resolve against the settings file's directory and are frozen as absolute paths when preparing a run.

| Field | Meaning |
|---|---|
| `paths.reports` | Directory containing `Lx_Ny/` activation-only report packages |
| `paths.activations` | Canonical saved activation archive |
| `paths.model` | Local Hugging Face dense-Qwen3 model directory |
| `paths.tokenizer` | Optional tokenizer JSON override; defaults to `model/tokenizer.json` |
| `paths.state` | Host locks and Unix sockets; use a short local path |
| `paths.references` | Optional local original-page snapshots named `Lx_Ny.html`, used only at publication |
| `gpu_groups` | Physical GPU groups; default `[[0,1,2,3]]` |
| `execution.image` | Built agent image name |
| `execution.start_next_after` | Default empty string: start all neurons; interpretation and revision can overlap. Set `02-experiments` to delay the next neuron until that handoff. |
| `spectrum_sampling` | Default 500 shared positions, label batches of at most 256, two concurrent labeling requests |
| `llm` | Provider/model/reasoning configuration, output capacity and upstream URLs |
| `pricing` | Explicit token-price estimate and peak-hour schedule |

Scientific settings (sampling, atlas, family, graph and blind-test parameters) live in the same file. Host paths, pricing and scheduling are not included in agent-facing scientific settings. The default research agent uses DeepSeek with high reasoning effort. Set the output capacity to the maximum supported by your endpoint.

Credentials stay on the host. Set `DEEPSEEK_API_KEY` in your shell environment, or set `NEURON_PAPER_API_KEY_FILE` to a local key file. Never put a key in settings or in Git. The agent receives only a temporary credential for the host API proxy. Authentication headers are not written to the API request records.

Set `NEURON_PAPER_BUDGET_USD` **when preparing** a batch. The default is USD 2 for the batch. The meter reserves conservative upper bounds for in-flight requests and preserves uncertain-charge upper bounds after interruption. Estimates use the configured tariff, not the account invoice. A resumed batch reuses its existing ledger. Separate batches do not automatically share one ledger.

## Start, continue and publish

```bash
NEURON_PAPER_BUDGET_USD=5 ./run.sh prepare \
  --settings settings.local.json --output runs/example \
  --neurons L55_N11703 L1_N20535
./run.sh execute --run runs/example
```

`prepare` saves a snapshot of the runtime source and configuration. `execute` uses that snapshot, so later edits do not change the prepared run. Preparation, each stage and the API meter save their records under the run directory.

Calling `execute` again skips stages already marked complete. An interrupted stage retains its workspace and measurements; inspect its records before continuing. To start from a saved handoff in a new run:

```bash
./run.sh prepare --settings settings.local.json --output runs/revision \
  --from revise --handoff /absolute/path/to/02-experiments/handoff \
  --neurons L55_N11703
```

For an existing-data experiment stage, use `--reuse-measurements HANDOFF_DIRECTORY`. Read the skill's existing-data workflow for what it can produce without fresh model inference. `--through interpret` stops after the initial explanation.

Publish a completed report to a new directory:

```bash
./run.sh publish --run runs/example --neuron L55_N11703 --output results/example
python -m http.server 8000 --directory results/example
```

Publication preserves the generated scientific content and adds shared display controls. If `paths.references/Lx_Ny.html` exists, the export includes a local original-page snapshot and a collapsible Compare control. Its two panes align corresponding sections and subsections, link scrolling, and support a resizable divider. The reference page is never mounted into an agent or included in a scientific handoff. Without a reference snapshot, publication exports the independent report without that control.

Top, Bottom, quantile and cluster examples initially show the selected anchor with up to 32 preceding tokens; expand an example to read its saved context. Activation spectrum examples retain up to 128 saved tokens before the first highlighted activation. Display windows do not change model inputs or scoring positions.

During the experiment and revision stages, `scripts/read_evidence.py --section brief` combines section summaries, experiment paths and prediction feedback. `--sections SECTION ...` reads several complete evidence sections in one call. After a final submission, `scripts/render_report.py` in the revision skill can update captions, introductions and displayed figure selections without repeating review or response prediction; see its [API](../skills/revise/references/api-reference.md#update-captions-and-presentation).

## Records

- `manifest.json`: frozen configuration, units, paths and budget.
- `framework-source/`: runtime source used by this run.
- `logs/api/`: each complete outgoing request, streamed response, usage and billing estimate.
- `logs/usage.json`: cumulative estimates and in-flight/unknown reservations.
- `units/Lx_Ny/01-interpretation/preparation/spectrum-sample/`: sampled positions, exact contexts, bin quotas and corpus counts.
- `units/Lx_Ny/01-interpretation/preparation/initial-scan/`: saved-data preparation timing.
- `units/Lx_Ny/*/workspace/response-predictions-*/`: frozen explanations and per-sample predictions joined to measured responses.
- `units/Lx_Ny/*/logs/agent/`: harness events and terminal status.
- `units/Lx_Ny/*/workspace/`: tool inputs, outputs and full spilled shell results.
- `units/Lx_Ny/*/handoff/`: JSON, HTML and measured artifacts for the next stage.
- `units/Lx_Ny/PIPELINE_TERMINAL.json`: final state and report path.
- `BATCH_TERMINAL.json`: per-neuron terminal states.

## Reproduce the published sampling and scheduling

Start from `settings.json`, which contains the shared 500-point sampling defaults, then configure your own model, activation archive and initial report paths. The [data guide](data.md#corpora-used-in-the-comparisons) identifies the published inputs; exact archive downloads are not yet available.

The recorded investigations used `deepseek-v4.1-flash-1017` with high reasoning. Set `llm.model` and the API URLs to the identifiers supported by your endpoint. A provider alias such as the default `deepseek-flash` can change over time and does not identify the recorded model version by itself.

For two independent four-GPU lanes on one Linux host, make two local copies of your configured settings file. Set `gpu_groups` to `[[0,1,2,3]]` in `settings.lane0.local.json` and `[[4,5,6,7]]` in `settings.lane1.local.json`. Both files must resolve `paths.state` to the same host directory. Keep `execution.start_next_after` empty. The following commands use the ten formal-paper neurons from the published SlimPajama comparison:

```bash
NEURON_PAPER_BUDGET_USD=unlimited ./run.sh prepare \
  --settings settings.lane0.local.json --output runs/lane0 \
  --neurons L55_N11703 L22_N19087 L52_N24188 L57_N16430 L15_N10957
NEURON_PAPER_BUDGET_USD=unlimited ./run.sh prepare \
  --settings settings.lane1.local.json --output runs/lane1 \
  --neurons L57_N2065 L50_N3507 L24_N17907 L51_N20229 L31_N23558
./run.sh execute --run runs/lane0 &
lane0_pid=$!
./run.sh execute --run runs/lane1 &
lane1_pid=$!
wait "$lane0_pid"
wait "$lane1_pid"
```

Each lane starts its interpretation stages concurrently. A host file lock covers the whole experiment stage, including its handoff, and is released before revision. The default lock name is derived from the configured physical GPUs under `paths.state/gpu-locks/`. Use one fixed, non-overlapping GPU group per lane. Batches sharing the same group and state directory share the same lock. `NEURON_PAPER_STAGE2_LOCK=/absolute/path/to/shared.lock` can explicitly select a common stage lock when preparing batches. The worker-level GPU lock and active-process check also remain in effect for each measurement call.

Set a finite `NEURON_PAPER_BUDGET_USD` before preparation if required; `unlimited` matches the recorded unrestricted run policy and permits ongoing API charges. Each batch saves its full configuration and records independently. Agent containers use an 8 GiB memory limit and a 6 GiB JavaScript heap limit.
