# Input data

The pipeline takes **saved native activations** and local model weights as inputs. Activation archives can contain any number of tokens, provided they follow the format below.

## A target report package

`paths.reports/Lx_Ny/` contains:

| File | Contents |
|---|---|
| `index.html` | Activation-only report using the bundled renderer and JSON bundle |
| `report.json` | Identity, population statistics, selected positions, full saved documents and three-channel traces |
| `token-rankings.json` | Six token-selectivity tables, generated from saved activations |
| `dataset-index.json` | Opaque report case IDs mapped to canonical document/token positions |
| `effects.json` | Optional existing output-effect measurements |
| `peak-census.json`, `peak-census-negative.json` | Optional saved censuses; otherwise derived from saved token extrema |

`report.json` includes `identity` (`id`, `layer`, `neuron`, `model`), `statistics`, `methods`, `examples`, `bands`, `scatter`, `points` and `documents`. Each point identifies its document, token position, and `[gate, value, product]`. Each document contains exact `token_ids`, decoded `tokens`, and aligned `gate`, `value`, `product` arrays. Cases are report-local identifiers; their names are not global token offsets.

The initial stage receives only the target's measurement files and activation-only HTML. Its inputs contain native statistics, extrema and quantile examples. Scan points remain private until the initial explanation is submitted, when feedback samples receive predictions from the frozen explanation. API records and the full activation archive remain on the host.

## Canonical activation archive

The bundled adapter is `tools/dataset_reader.py`. A dataset-local `reader.py` can implement the same interface and is treated as trusted executable input.

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

The scan uses `histogram`, `iter_neuron`, document rows and token IDs. The optional extremes/statistics methods use the additional index files. A custom adapter exposes `manifest`, `neurons`, `parts`, `locations`, `documents`, `starts`, `token_ids`, `document()`, `document_activations()`, `iter_neuron()`, `histogram()` and `close()`.

## Generate initial reports from an existing archive

`analysis/build_dataset_reports.py` builds initial reports from saved activations. It expects the original token-input sidecars (`block-NNNNNN.ids.u32.npy`, `.offsets.u64.npy`, `.documents.json`) and capture metadata in the archive manifest: `model`, `model_snapshot`, `capture`, `selection`, `documents` and `neuron_count`.

```bash
python analysis/build_dataset_reports.py \
  --dataset /path/to/archive --inputs /path/to/input-sidecars \
  --tokenizer /path/to/model/tokenizer.json \
  --existing /path/to/existing-reports --output /path/to/new-reports \
  --workers 2 --seed 42
```

Omit `--existing` if there are no reports to reuse. This is a CPU saved-data scan. Add token tables to copies with the separate saved-data command:

```bash
python analysis/add_token_tables.py --help
```

Alternatively the report builder accepts `--token-tables --token-cache DIR` to reuse the token-count/label cache in the same scan. The full argument list is available with `--help`.

## Coordinates and sampling

The measured activation is `SiLU(gate) * up`, taken from the native model tensor rather than reconstructed from rounded display values. Scatter point colours show blind predictions from a frozen explanation: orange for positive, blue for negative and grey for uncertainty. Stored records keep the prediction, measured activation and explanation version separately.

The pre-interpretation scan samples at most 40 candidates per activation-amplitude interval. It then retains up to 256 points in proportion to **candidate-bin counts**, with largest-remainder allocation and a fixed seed. The selected scan points are kept private until the initial explanation is submitted. A document-disjoint portion remains private until the revised explanation is submitted. These are not proportions of the full corpus and the plot is not a density estimate. Extrema and other measured experimental points are retained by their separate selection rules.

All saved contexts used for model experiments preserve the captured prefix through the chosen anchor. The display can show a shorter window without changing the model input or scoring position.
