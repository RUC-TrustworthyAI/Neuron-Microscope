# Hosting

The exported viewer works on any static file server. For large galleries, `hosting/index.mjs` adds a same-origin R2 media service: static HTML stays small, while images and node records live in object storage.

Run the following commands from `vision/`, with the vision environment activated.

## 1. Export the viewer

```bash
vision-microscope export \
  --sample-run vision-repro/outputs/imagenet_scan \
  --published-api \
  --output vision-repro/outputs/site
```

Optionally add `--runs`, `--rf-snapshot`, and `--reference-catalog` as in the [running guide](reproduction.md). Host at the domain root. Configure the Worker with:

- `ASSETS`: the exported static directory, including original sample atlases.
- `BUCKET`: its R2 bucket.
- `/api/*` and `/media/*` routed to the Worker before static assets.

`hosting/index.mjs` delegates other requests to `ASSETS`. The frontend retains bundled results when no cloud release is available. No browser credentials are needed to view published images.

## 2. Prepare immutable snapshots

For natural examples, use the RF lossless or preview snapshot from `crop-examples`.
For a completed or partially completed A/1024 campaign:

```bash
python -m vision_repro.prepare_site_snapshot \
  vision-repro/outputs/all_neurons vision-repro/outputs/optimized_release
```

The exporter checks recorded source hashes, completion markers, dimensions and activation scores. A positive result must have positive activation; a negative result must have negative signed activation. All four Diversity members must pass. Failed checks appear as `quality_review`, not completed images. Original optimization artifacts stay in the campaign directory.

Use a unique output name containing letters, digits, `_` or `-`; it becomes the immutable release ID.

## 3. Upload and activate

Using the hosting provider's secret manager, configure a temporary random 64-character `IMPORT_TOKEN` and an ISO-8601 `IMPORT_EXPIRES_AT` timestamp. Deploy that environment, then run:

```bash
python -m vision_repro.publish_site_snapshot \
  vision-repro/outputs/rf_preview --origin https://your-microscope.example --workers 8
```

The hidden prompt accepts a JSON object with `import` set to the temporary token. A deployment that requires a Sites service authorization header can also supply `service`. Credentials stay in process memory.

Uploads are bounded, resumable and protected by an exclusive local lock. `upload-ack.jsonl` records verified object receipts for this origin. Every object is checked again before activating the release; the viewer changes only after the complete snapshot passes. `--available-only` can transfer a partial local copy without activating it.

After publication, remove both import variables from the hosting environment and redeploy. Public reads continue to work. `publication.json` records the active release, verified counts and bytes.

## Routes

| Route | Purpose |
|---|---|
| `/api/published` | Current optimized-image release |
| `/api/catalog/<release>` | Optimized-node availability |
| `/api/neuron/<release>/<layer>/<channel>` | Five-mode results and metadata |
| `/api/examples/published` | Current natural-example release |
| `/api/examples/<release>/<layer>/<channel>` | RF crops, rankings and heatmaps |
| `/media/<sha256>.webp` | Immutable image |

Optimized images and natural examples use separate `active.json` and `examples-active.json` pointers. Publishing one leaves the other in place. The source repository contains implementation and demonstration figures; datasets and generated galleries live outside Git.

Run API checks with `node hosting/test-media.mjs` and viewer checks with `node vision-repro/tests/test_viewer.mjs`.
