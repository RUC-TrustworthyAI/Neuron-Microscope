# Running

## Host and container

Use Linux and Bash 4+ for the production shell orchestration. The offline demo can also run on macOS. Python must be 3.10 or newer.

The host reads saved arrays, serves evidence and runs GPU workers. Containers run the agent only; installing CUDA in the agent image is unnecessary. Each container receives a single skill and an isolated workspace. Inputs and handoffs are read-only mounts. The first agent receives only its own report directory; full activations and weights are not mounted there. Host preparation uses the archive and tokenizer before that agent starts.

Install host dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/gpu.txt
# Choose a CUDA wheel compatible with your NVIDIA driver.
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

The recorded development environment used torch 2.7.1+cu118, Transformers 4.57.6, tokenizers 0.22.2, NumPy 2.2.6, PyArrow 21.0.0 and SciPy 1.15.3. The adapter shards complete dense-Qwen3 layers over the configured GPU group. Available memory depends on model size, input prefixes and backward storage; changing the group does not change the model identity.

Build the agent image:

```bash
docker build -t neuron-investigator:local .
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
| `execution.start_next_after` | Default `02-experiments`: next neuron starts after that handoff |
| `llm` | Provider/model/reasoning configuration, output capacity and upstream URLs |
| `pricing` | Explicit token-price estimate and peak-hour schedule |

Scientific settings (sampling, atlas, family, graph and blind-test parameters) live in the same file. Host paths, pricing and scheduling are not included in agent-facing scientific settings. The default investigator uses DeepSeek with high reasoning effort. Set the output capacity to the maximum supported by your endpoint.

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

Top, Bottom, quantile samples, and cluster examples share the display rule: retain up to 128 saved tokens before the first token highlighted by the existing colour rule, while preserving the scored anchor and the remaining saved context. This does not shorten model inputs or change measurements.

## Records

- `manifest.json`: frozen configuration, units, paths and budget.
- `framework-source/`: runtime source used by this run.
- `logs/api/`: each complete outgoing request, streamed response, usage and billing estimate.
- `logs/usage.json`: cumulative estimates and in-flight/unknown reservations.
- `units/Lx_Ny/01-interpretation/preparation/initial-scan/`: scan and semantic labeling.
- `units/Lx_Ny/*/logs/agent/`: harness events and terminal status.
- `units/Lx_Ny/*/workspace/`: tool inputs, outputs and full spilled shell results.
- `units/Lx_Ny/*/handoff/`: JSON, HTML and measured artifacts for the next stage.
- `units/Lx_Ny/PIPELINE_TERMINAL.json`: final state and report path.
- `BATCH_TERMINAL.json`: per-neuron terminal states.
