# Structured workflow

This entry point investigates a neuron through direct, schema-constrained LLM calls. Python selects saved evidence, executes measurements, joins labels to observations and renders the existing interactive report. Prompts and output contracts are in `workflow/prompts.py` and `workflow/contracts.py`.

## Run

Use the Linux GPU environment described in [Running](running.md#environment), a local dense-Qwen3 model, a grouped activation archive and an activation-only report package described in [Data](data.md). The report must contain measured static logit effects in `effects.json`.

Create a configuration from the repository settings, adjusting the paths and GPU indices:

```python
import json
from pathlib import Path

base = json.loads(Path('settings.json').read_text())
base['gpu_groups'] = [[0, 1, 2, 3]]
base['llm']['model'] = 'deepseek-v4.1-flash-1017'
base['llm']['reasoning_effort'] = 'high'
base['spectrum_sampling'] = {'total_points': 500}
config = {
    'neuron': 'L55_N11703',
    'model': str(Path('models/subject').resolve()),
    'report': str(Path('data/reports/L55_N11703').resolve()),
    'activations': str(Path('data/activations').resolve()),
    'gpus': [0, 1, 2, 3],
    'settings': base,
    'llm': dict(base['llm'], pricing=base['pricing']),
    'api_slots': str(Path('.local/api-slots').resolve()),
    'gpu_lock': str(Path('.local/gpu-0123.lock').resolve()),
    'search_rounds': 3,
    'search_regions': 6,
    'candidates_per_region': 16,
    'atlas_natural': 20,
    'atlas_written': 20,
    'batch_size': 8,
    'batch_tokens': 8192,
}
Path('workflow.local.json').write_text(json.dumps(config, indent=2))
```

Configure the API URLs for an endpoint that provides the selected model. Set `DEEPSEEK_API_KEY` in the environment or `NEURON_PAPER_API_KEY_FILE` to a local credential file, then run:

```bash
python -m workflow.run --config workflow.local.json --output runs/example
python -m http.server 8000 --directory runs/example/report
```

To inspect the explanation and both labelled plots before constructing experiments, append `--stop-after interpretation`. This uses saved activations and API calls without loading the target model. An optional `saved_sample` configuration path reuses an existing workflow’s `sample/` directory, preserving its sampled positions and contexts.

`max_rules_per_pole` sets the maximum number of positive rules and negative rules in both the initial explanation and its single revision. It defaults to 3. `No visible cue` is supplied separately.

`batch_tokens` limits padded tokens per measurement batch. The model remains loaded throughout the measurement stage. Runs sharing a physical GPU group must use the same `gpu_lock`; runs sharing an API concurrency allowance on one host must use the same `api_slots` directory. Separate GPU groups can measure different neurons concurrently. The API concurrency limit is `llm.max_concurrent_requests`.

Optional `publish`, `references` and `publish_neurons` fields export reports and a comparison directory. `references` contains snapshots named `Lx_Ny.html`; they are read only after research is complete. Each investigation writes to a new directory. Repeating the command with the same configuration reuses completed API responses, sampling and measurement files. Change the output directory when changing scientific inputs.

## Measurements and interpretation

1. Sample 500 saved positions across activation bins, with allocation proportional to the displayed log-count heights. Give the explanation model all sampled contexts and their measured gate, up and product values. Shared document text is included once.
2. Request an initial explanation and up to three response rules per pole. Predict the shared 500 positions without their measured activations, using two batches of at most 256. Missing predictions are requested again; when fewer than 16 remain, both charts show the same successfully labelled positions. Disagreement with measured signs remains visible. Refine the explanation and rules once using the measured contexts and prediction feedback, then repeat blind labelling and freeze the result before constructing experiments.
3. Run activation search with measured feedback, ten-condition controlled tests, blind written probes, local attribution, family traces and interventions. Search measures full-input maxima or minima. Controlled tests use the observed search results to design positive, negative and neutral conditions, specifying a replacement span or a common suffix. They measure the signed largest absolute response in that span, or a single specified token. Every chart condition has a short label and a separate complete input text. The headline uses the common-frame test with the strongest coverage of both corpus poles.
4. Analyze the measured experiments against the frozen explanation. Keep contrary results in criticism and section prose. An independent reader assesses the account, its supported branches and unresolved patterns. The two plots retain the frozen rules and final blind predictions. These samples provide development feedback and are not a held-out evaluation.
5. Assemble the HTML from the structured outputs and measured values. Static Promotes/Suppresses values come from the report's measured logit effects. Semantic clusters retain the measured atlas cases, including unresolved cases.

## Records

`status.json` reports the current step. `api/` retains requests, raw streams, reasoning, outputs, usage, cost estimates and timing. `measurements/`, `atlas/` and `graphs/` retain measured data; `artifacts.json` supplies the renderer. No authentication headers are stored.

GPU time counts recorded model operations and loading. API streaming time counts overlapping streams once. Execution time includes CPU work, API latency and GPU-lock waits within the running process, excluding gaps between resumed processes. These intervals can overlap, so GPU and API times do not sum to execution time. Activation collection is outside the report run. Prices are configured estimates; usage records are retained for comparison with the provider invoice.
