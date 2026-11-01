# Structured workflow

This entry point investigates a neuron through direct, schema-constrained LLM calls. Python selects saved evidence, executes measurements, joins labels to observations and renders the existing interactive report. Prompts and output contracts are in `language/workflow/prompts.py` and `language/workflow/contracts.py`.

## Run

Use the Linux GPU environment described in [Running](running.md#environment), a local dense-Qwen3 model, a grouped activation archive and an activation-only report package described in [Data](data.md). The report must contain measured static logit effects in `effects.json`.

Create a configuration from the repository settings, adjusting the paths and GPU indices:

```bash
python language/examples/configure.py \
  --neuron L55_N11703 \
  --model /path/to/model \
  --report /path/to/activation-only/report \
  --activations /path/to/activation/archive \
  --gpus 0 1 2 3 \
  --output workflow.local.json
```

The generator reads `language/settings.json`, using `deepseek-flash` with `high` reasoning and configurable endpoint URLs and token prices. The reports record the model as `deepseek-v4.1-flash-1017`. Review these settings for your endpoint. Set `DEEPSEEK_API_KEY` in the environment or `NEURON_PAPER_API_KEY_FILE` to a local credential file, then run:

```bash
python -m language --config workflow.local.json --output runs/example
python -m http.server 8000 --directory runs/example/report
```

To inspect the explanation and both labelled plots before constructing experiments, append `--stop-after interpretation`. This uses saved activations and API calls without loading the target model. An optional `saved_sample` configuration path reuses an existing workflow’s `sample/` directory, preserving its sampled positions and contexts.

`max_rules_per_pole` sets the per-pole rule limit. The generator sets it to 5 and `max_refinements` to 2, producing an initial explanation, two revisions and three labeling passes. A hand-written configuration that omits the rule limit uses 3. `No visible cue` is supplied separately; a pole without supported rules may remain unexplained.

`batch_tokens` limits padded tokens per measurement batch. Set `NEURON_MICROSCOPE_GPU_WORKER=1` to share one resident model per GPU group across queued operations from different runs; API requests can continue while other runs use the GPU. Runs sharing a physical GPU group must use the same `gpu_lock`. Without the worker, the lock covers a run’s complete measurement stage. Separate groups can measure concurrently. `llm.max_concurrent_requests: null` imposes no application-level API cap; a numeric limit uses the shared `api_slots` directory.

Optional `publish`, `references` and `publish_neurons` fields export reports and a comparison directory. `references` contains snapshots named `Lx_Ny.html`; they are read only after research is complete. Each investigation writes to a new directory. Repeating the command with the same configuration reuses completed API responses, sampling and measurement files. Change the output directory when changing scientific inputs.

## Measurements and interpretation

1. Divide 500 positions equally among nonempty activation bins. If a bin has fewer saved positions than its quota, redistribute the shortfall. Select evenly spaced ranks within each bin’s complete corpus population. The explanation model receives product values and contextual text for the full sample, plus extreme tokens, document exemplars, quantile cases, token selectivity, peak statistics and direct logit effects. Gate and up measurements remain in the archive and plots but are excluded from explanation inputs. Shared document text is included once.
2. Request an initial explanation with up to five response rules per pole. Label the shared 500 positions with measured activations hidden, using five concurrent batches of at most 100. Revise twice, each time reading the preceding explanation, all 500 measured contexts and focused feedback: sign reversals and strong responses above 50% of the corresponding corpus extreme assigned to No visible cue. Label each revised version. Among versions with directional accuracy >90%, choose the highest coverage; if none qualifies, choose the highest accuracy. Freeze the selected explanation with its corresponding labels. Missing predictions are requested again until fewer than 16 remain; both plots retain the same successfully labeled positions and keep visible disagreements.
3. Run three Search rounds with measured feedback, requesting six regions and 16 texts per region. Design four controlled tests with ten conditions each, then improve the test design once using measurements and retain the stronger version of each test. Conditions use a replacement span or common suffix, scoring the signed largest absolute response or a specified single token. The head figure selects the test with the highest positive response when positive rules exist, otherwise the strongest negative response. Its labels remain separate from the complete inputs. The independent Blind writer sees only the frozen claim and rules and writes six texts per supported pole; Search and Blind score full-input maxima or minima. Measure 20 natural and 20 written attribution cases, family responses, three graphs and matched interventions.
4. Analyze the measured experiments against the frozen explanation. Keep contrary results in criticism and section prose. An independent reader assesses the account, its supported branches and unresolved patterns. The two plots retain the frozen rules and final blind predictions. These samples provide development feedback and are not a held-out evaluation.
5. Assemble the HTML from the structured outputs and measured values. Static Promotes/Suppresses values come from the report's measured logit effects. Semantic clusters retain the measured atlas cases, including unresolved cases.

## Records

`status.json` reports the current step. `api/` retains requests, raw streams, reasoning, outputs, usage, cost estimates and timing. `measurements/`, `atlas/` and `graphs/` retain measured data; `artifacts.json` supplies the renderer. No authentication headers are stored.

GPU time counts recorded model operations and loading. API streaming time counts overlapping streams once. Execution time includes CPU work, API latency and GPU-lock waits within the running process, excluding gaps between resumed processes. These intervals can overlap, so GPU and API times do not sum to execution time. Activation collection is outside the report run. Prices are configured estimates; usage records are retained for comparison with the provider invoice.
