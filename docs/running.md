# Running

## Environment

Use Linux and Python 3.10+ for an investigation. The offline renderer also runs on macOS. Full measurements require a CUDA-capable host and local dense-Qwen3 weights.

```bash
python3 -m venv .venv
source .venv/bin/activate
# Choose a CUDA wheel compatible with your NVIDIA driver.
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements/gpu.txt
```

The measurement adapter shards complete dense-Qwen3 layers over the configured GPU group. Required memory depends on model size, captured prefixes and backward storage.

## Configure and run

Create `workflow.local.json` from the [configuration example](workflow.md#run). Supply a local model, a canonical activation archive and a target activation-only report package as described in [Input data](data.md). The report package must contain measured static logit effects in `effects.json`.

The configuration selects the neuron, GPU indices, shared GPU lock, shared API slot directory, sampling settings and measurement batch sizes. `llm` specifies the endpoint, model, reasoning effort, output capacity, concurrency and token pricing. To use the recorded API model, configure `deepseek-v4.1-flash-1017` with `high` reasoning and an endpoint that provides that identifier.

Set `DEEPSEEK_API_KEY` in the environment or point `NEURON_PAPER_API_KEY_FILE` to a local credential file. Keep credentials out of configuration files and Git. API records omit authentication headers.

```bash
python -m workflow.run --config workflow.local.json --output runs/example
python -m http.server 8000 --directory runs/example/report
```

To inspect the refined explanation and both labeled plots before experiments:

```bash
python -m workflow.run --config workflow.local.json --output runs/example \
  --stop-after interpretation
```

This step uses saved activations and LLM requests without loading the target model. Running the full command afterward continues with the saved explanation and predictions.

## Concurrency and continuation

Launch each neuron with its own configuration and output directory. Runs on the same physical GPU group must use the same `gpu_lock` path. The lock covers loading the target model and the complete measurement phase; the model stays loaded throughout that phase. Use disjoint groups to measure neurons concurrently. Explanation and report-analysis requests can overlap across runs.

Runs sharing an API concurrency allowance on one host must use the same `api_slots` directory and `llm.max_concurrent_requests` value. Labeling uses two concurrent batches of at most 256 positions within that allowance.

A run saves `config.json` and rejects a different configuration in the same output directory. Repeating a command with the same configuration reuses completed API responses, sampling files and available measurement records. Failed requests and partial labels remain saved for inspection and continuation. An optional `saved_sample` points to a workflow's `sample/` directory to reuse its exact positions and contexts. Use a new output directory when changing scientific inputs.

## Export a comparison

Optional configuration fields `publish`, `references` and `publish_neurons` export reports and their directory. `references` points to local original-page snapshots named `Lx_Ny.html`; the publisher reads them only after research is complete. The published export opens with the reference on the left and reproduction on the right. Either pane can control synchronized scrolling. Close the comparison or append `?compare=0` to read the reproduction alone.

The [report guide](reports.md) explains display windows and section contents. Display windows leave the measured model inputs and scoring positions unchanged.

## Records

| Path within a run | Contents |
|---|---|
| `config.json` | The run's input and execution settings |
| `status.json`, `failure.json` when present | Current phase and failure details |
| `sample/`, `evidence.json` | Sampled positions, saved contexts and measured interpretation input |
| `initial-labels/`, `refined-labels/` | Independent predictions and partial-label checkpoints |
| `interpretation.json`, `predictions.json` | Refined explanation and frozen final predictions |
| `api/` | Complete requests, raw streams, reasoning, structured outputs, usage, estimated cost and timing |
| `measurements/`, `atlas/`, `graphs/` | Model measurements and attribution results |
| `gpu-queue.json` | GPU wait duration and assigned devices |
| `artifacts.json`, `report/` | Renderer input and interactive report |

GPU time includes recorded model operations and loading. API streaming time counts overlapping streams once. Execution time includes CPU work, API latency and GPU-lock waits within the running process, excluding gaps between resumed processes. These intervals can overlap. Earlier activation collection is outside the report run. Token costs use configured prices and can be checked against the provider's invoice.
