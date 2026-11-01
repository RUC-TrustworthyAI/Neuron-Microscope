# Input data

The investigation workflow reads **saved native activations** and local model weights. The collection command below creates the archive from local text; its token budget is configurable.

## Collect activations from local text

Run `python -m language.collection --help` for the collection entry point. It accepts local JSONL and Parquet files with a string column, a local dense-Qwen3 model, target neuron coordinates and a token budget. The [Usage example](../../README.md#2-collect-activation-evidence) collects 100M tokens and then configures an investigation using its outputs.

Documents are permuted with a fixed seed. The collector uses the first `--max-document-tokens` tokens of each selected document, without added special tokens, and trims the final prefix to the remaining budget. Each document has its own causal context. Parquet text is temporarily cached in memory-mapped Arrow files for random access; the cache is removed after token preparation.

The default prefix limit is 4,096 tokens. `--batch-size` defaults to 4, `--batch-tokens` to 8,192 padded tokens and `--shard-tokens` to 262,144. Whole document prefixes stay within one shard. `--tokens` may be reduced for a small run. The model runs in BF16 with SDPA; hooks capture raw gate, raw up/value and the native input to `down_proj`.

The output contains:

```text
collection/
  dataset.json, neurons.json, verification.json
  token_ids.npy, documents.parquet
  inputs/block-NNNNNN.ids.u32.npy
  inputs/block-NNNNNN.offsets.u64.npy
  inputs/block-NNNNNN.documents.json
  part-0/shard-NNNNNN.npy
  part-0/histogram.npy
  indices/extremes.npz, indices/statistics.parquet
  effects/<neuron-id>.json
  reports/<neuron-id>/
```

`reports/<neuron-id>/` contains the files listed below, including `effects.json` and `token-rankings.json`. Use the collection directory as `--activations` and the neuron's report directory as `--report` when creating a workflow configuration. The command uses the existing saved-data renderer for report construction. Repeating it with the same inputs resumes committed shards and completed reports. `status.json` records progress.

## A target report package

The workflow configuration’s `report` path points to a target package containing:

| File | Contents |
|---|---|
| `index.html` | Activation-only report using the bundled renderer and JSON bundle |
| `report.json` | Identity, population statistics, selected positions, full saved documents and three-channel traces |
| `token-rankings.json` | Six token-selectivity tables, generated from saved activations |
| `dataset-index.json` | Opaque report case IDs mapped to canonical document/token positions |
| `effects.json` | Measured static logit effects used by Promotes/Suppresses |
| `peak-census.json`, `peak-census-negative.json` | Optional saved censuses for the corresponding report sections |

`report.json` includes `identity` (`id`, `layer`, `neuron`, `model`), `statistics`, `methods`, `examples`, `bands`, `scatter`, `points` and `documents`. Each point identifies its document, token position, and `[gate, value, product]`. Each document contains exact `token_ids`, decoded `tokens`, and aligned `gate`, `value`, `product` arrays. Cases are report-local identifiers; their names are not global token offsets.

The explanation model receives sampled contexts with measured product values, together with extreme tokens, document exemplars, quantile cases, token selectivity, peak statistics and direct logit effects. Gate and up channels remain saved but are excluded from explanation requests. Blind labeling hides all measured activations. Two explanation revisions and three labeling passes precede version selection and freezing. Official explanations and previous investigation conclusions are excluded from research inputs.

## Canonical activation archive

`language/tools/dataset_reader.py` reads the archive format below. The interpretation workflow's sampling step uses this built-in reader.

```text
archive/
  dataset.json
  neurons.json
  documents.parquet
  token_ids.npy
  indices/...
  part-0/
    shard-000000.npy
    histogram.npy
```

A manifest references paths instead of depending on those example names. Its fields used by the adapter are:

```json
{
  "complete": true,
  "tokens": 1024,
  "neurons": "neurons.json",
  "channels": ["gate", "value", "product"],
  "corpus": {"documents": "documents.parquet", "token_ids": "token_ids.npy"},
  "parts": [{
    "id": "part-0", "path": "part-0",
    "shards": [{"id": 0, "path": "shard-000000.npy", "token_start": 0, "token_count": 1024}],
    "indices": {"histogram": "histogram.npy"}
  }],
  "indices": {"k": 30, "extremes": "indices/extremes.npz", "statistics": "indices/statistics.parquet"}
}
```

- `neurons.json` is a list with `id`, `layer`, `neuron`, `part` and `part_column` per neuron.
- Document rows, sorted by `document_id` and corpus order, have `document_id`, `token_start`, `token_count`, `shard_id` and `shard_token_start`.
- Token IDs are a one-dimensional unsigned integer NPY array.
- Each activation shard is **neuron × token × channel**, stored as the raw BF16 bit patterns in `uint16` NPY arrays. These are not numeric uint16 activation values.
- Channel order is raw gate preactivation, raw up/value preactivation, and the captured native product at the input of `down_proj`.
- Histograms have shape `[part_neurons, 3, 65536]` and `uint64` counts, one bin for each BF16 bit pattern.
- Documents must retain the same causal input prefix used during collection. Document boundaries must align with shards for this adapter.

The scan uses `histogram`, `iter_neuron`, document rows and token IDs. The optional extremes/statistics methods use the additional index files. A custom adapter for the report builder exposes `manifest`, `neurons`, `parts`, `locations`, `documents`, `starts`, `token_ids`, `document()`, `document_activations()`, `iter_neuron()`, `histogram()` and `close()`.

## Generate initial reports from an existing archive

`language/analysis/build_dataset_reports.py` builds initial reports from saved activations. It expects the original token-input sidecars (`block-NNNNNN.ids.u32.npy`, `.offsets.u64.npy`, `.documents.json`) and capture metadata in the archive manifest: `model`, `model_snapshot`, `capture`, `selection`, `documents` and `neuron_count`.

The report builder loads `Dataset` from a dataset-local `reader.py` when present. Both this builder and `language/analysis/add_token_tables.py` require `dataset.json` and the archive's verification record, `verification.json`, to contain `complete: true`.

```bash
python language/analysis/build_dataset_reports.py \
  --dataset /path/to/archive --inputs /path/to/input-sidecars \
  --tokenizer /path/to/model/tokenizer.json \
  --existing /path/to/existing-reports --output /path/to/new-reports \
  --workers 2 --seed 42
```

Omit `--existing` if there are no reports to reuse. This is a CPU saved-data scan. Add token tables to copies with the separate saved-data command:

```bash
python language/analysis/add_token_tables.py --help
```

Alternatively the report builder accepts `--token-tables --token-cache DIR` to reuse the token-count/label cache in the same scan. The full argument list is available with `--help`.

## Coordinates and sampling

The measured activation is `SiLU(gate) * up`, taken from the native model tensor rather than reconstructed from rounded display values. Scatter point colours show blind predictions from a frozen explanation: orange for positive, blue for negative and grey for uncertainty. Stored records keep the prediction, measured activation and explanation version separately.

The workflow divides 500 positions equally among nonempty activation bins. If the complete corpus population of a bin cannot fill its quota, the remaining quota is shared among other nonempty bins. Positions are selected at evenly spaced ranks within each bin’s full corpus population. The explanation model sees the full sample with product measurements. For blind labeling, the predictor receives the full captured prefix through each anchor and up to ten following tokens, with measured responses hidden. Following tokens provide linguistic context and are outside the causal prefix at that anchor.

Activation spectrum and Inside the neuron share this sample pool. The initial explanation and two revisions each receive a blind labeling pass, with five concurrent requests of up to 100 positions. Among versions with directional accuracy >90%, select the highest coverage; if none qualifies, select the highest accuracy. Both plots use the selected version’s own predictions, frozen before experiments. Further requests fill missing predictions until fewer than 16 remain; these missing positions are excluded from both plots. An explicit uncertain answer stays in No visible cue. The full-corpus histogram retains all token counts. These samples provide development feedback; they do not estimate population density or serve as a held-out evaluation.

All saved contexts used for model experiments preserve the captured prefix through the chosen anchor. The display can show a shorter window without changing the model input or scoring position.

## Corpora used in the comparisons

These are our saved evaluation inputs, counted with the Qwen3-32B tokenizer. Their labels refer to the following samples. Source links provide upstream data; the exact sampled texts and activation archives do not yet have a public download.

| Report label | Tokens | Source and selection |
|---|---:|---|
| SlimPajama | 100,000,000 | [DKYoon/SlimPajama-6B](https://huggingface.co/datasets/DKYoon/SlimPajama-6B), revision `b5f90f419b7489cdba26fdbc8c022fcb5562f968`. Assign uniform random priorities to all 5,489,000 documents in its 48 train shards, seed `2026102901`; consume in priority order and trim only the final document to the budget. The sampling population is this public 6B subset. |
| FineWeb | 99,999,922 | [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb). Uniform document permutation from our saved 1B-token activation archive, seed `2026102905`, retaining 159,156 samples near the 100M boundary. Captured prefixes cut by a token limit were trimmed to the last detected sentence ending. Native activation bits were reused. |
| Common Pile | 100,000,067 | [common-pile/comma_v0.1_training_dataset](https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset), train. Complete documents taken from the beginning of our seeded, shuffled 500M-token pool until reaching 100M. This sample includes scientific text, code, books, patents, legal text, wikis and forums. |
| Reasoning Mix | 100,000,000 | [EleutherAI/reasoning-mix](https://huggingface.co/datasets/EleutherAI/reasoning-mix), revision `325463c60e4d283679279277c1ee2f2aebe87e95`. Uniform random permutation of all 11,693,047 Parquet rows, seed `42`, without replacement. Each document contributes at most its first 4,096 tokens; the final document may be trimmed. The selected 99,724 documents comprise 81,143 `fineweb`, 18,448 `numina` and 133 `stratos` records; these are document counts, not token proportions. |

SlimPajama's sampled token proportions are Common Crawl 52.94%, C4 26.87%, arXiv 5.13%, GitHub 4.20%, books 3.93%, Wikipedia 3.78% and StackExchange 3.14% (rounded). No content filter or source quota was applied to that draw. Common Pile's largest sampled components are USPTO 34.33%, Stack v2 educational code 14.90%, peS2o 9.47%, PubMed 7.62% and StackExchange 5.24%; this sample has its own source balance.

Raw documents can exceed a capture window. SlimPajama and Common Pile were split into causal windows of at most 4,096 tokens for activation collection, with source positions retained. FineWeb and Reasoning Mix used captured document prefixes. Reports therefore describe the model's response to those saved inputs. Reproduction requires the corresponding input windows and tokenizer, not just a dataset name and token budget. Source datasets retain their own licenses and attribution requirements.
