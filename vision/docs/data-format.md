# Data format and provenance

**English** · [简体中文](data-format_zh.md)

## Independent optimization batches

```text
run/
  job.json           Fixed layer, channels, variant and configuration
  metadata.json      Model identity, versions, branch mapping and objective
  evaluation.json    Initial/checkpoint/final scores with the same definition
  history.json       Optimization scores, losses, frequency band, blur and LR
  complete.json      Completion, steps, elapsed time and allocator telemetry
  resume.pt          Fourier, Adam and RNG state, retained in the source run
  items/0/           First channel's initial/checkpoint/final PNGs
  items/1/           Next channel, using the batch's seed configuration
```

Each Diversity item contains four member images and a tiled preview. Evaluation score arrays correspond to those members. The viewer displays their mean and retains the full array in JSON.

Campaign batches use the same layout under `batches/<job>/`. `queue.sqlite` records states such as `pending`, `running`, `done`, `unavailable`, `failed` and `split`. `protocol.json` records the configuration, source digests and batching policy. `progress.json` and `telemetry.jsonl` support monitoring.

## Natural examples

A scan retains `inputs.npy`, layer scores and locations, `selected_images.jsonl`, and its protocol. Its `web/` directory contains:

- `neurons/<layer>/<channel>.json`: sample IDs, classes, ranks, scores, atlas positions, heatmap offsets/shapes and peaks for Top 16 and Additional Examples.
- `atlases/*.webp`: display atlases with eight images per row.
- `heat/<layer>/<channel>.bin`: uint8 heatmaps indexed by the offsets in JSON.
- `manifest.json`: relative asset paths and SHA-256 values.
- `complete.json`: scan, channel and example counts, plus the digest of the exact manifest bytes.

Heatmap intensities are normalized per image. Negative heatmaps display `max(-pre,0)`. They represent layer activations.

## Static export

`data.json` provides the 5,804-channel catalog, result paths and example sources. `our-results.json` uses `layer:channel` keys. `ours[variant][step]` points to PNGs; `variantInfo` retains configuration, initial/final scores, seed and model provenance. The exporter imports completed batches and leaves missing targets without result records.

An optional `reference-catalog.json` has this form:

```json
{
  "neurons": {
    "mixed4c:447": {"flags": "11111"}
  }
}
```

The five flags correspond to Positive, Path-guided +, Diversity, Negative and Path-guided −. `1` means verified available, `0` means unspecified, and `h` means confirmed unpublished on the reference site. Defaults are all `0`; mark availability only after verifying it. Reference URLs point to the official domain, and the exporter leaves those PNGs external.

Each export writes a fresh directory and asset digests. Generated sites stay outside source control. `public_metadata` removes common absolute local paths; review custom records for sensitive content before publication.

## Receptive-field snapshots

`crop-examples` creates an independent snapshot:

```text
rf_snapshot/
  examples/<release>/catalog.json
  examples/<release>/manifest.json
  examples/<release>/nodes/<layer>/<channel>.json
  images/<sha256>.webp
  upload-manifest.json
  complete.json
```

Each node records `cropPolicy`, the original `mapLocation`, `requestedRF`, actual `cropBounds`, and the `activationPosition` within the crop. `heatBase64` contains a 24 × 24 uint8 heatmap per example; `heatOffset` indexes its bytes. Negative heatmaps continue to use `max(-pre,0)`.

Shallow-layer crops use 8 × 6 atlases with 112 × 112 tiles. Deep full-image crops reuse 8 × 1 atlases. The viewer reads `atlasColumns` and `atlasRows` for placement. `atlasUrl` addresses an immutable image in cloud mode and becomes a relative local path in static exports.

Compressed snapshots retain `encoding-audit.json`. WebP quality starts at 95, with whole-atlas PSNR of at least 36 dB and per-tile PSNR of at least 30 dB; lossless encoding is retained when these thresholds are missed. Scores, ranks, heatmaps and crop coordinates stay unchanged.

`data.json.rfExamples` lists cropped nodes; other channels use the original scan assets. `publishedApi` defaults to false. When enabled, the viewer reads same-origin publication endpoints. Optimized images and natural examples use separate publication pointers.
