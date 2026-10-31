# Running and batch generation

**English** · [简体中文](reproduction_zh.md)

Run all commands from the repository's `vision/` directory, with the vision environment activated. Use Linux, Python 3.11+ and a PyTorch/CUDA build compatible with the target GPU. DGX Spark requires a build supporting its architecture. Provision the model as described in [model setup](../source/inceptionv1/README.md), then run `vision-microscope doctor`.

## 1. Check a single channel

```bash
vision-microscope optimize --layer mixed4c --neurons 447 \
  --variant positive --steps 8 --output vision-repro/outputs/smoke
```

An eight-step run checks pretrained model loading, gradients and saved records. Use 1,024 steps for the configured A experiment; the default seed is 0. Repeating the same command resumes from a checkpoint or reads a completed result. Use a new directory for a different configuration.

## 2. Retrieve natural examples

Obtain the ImageNet training archive under its access and usage terms. Keep the nested class-tar structure in `ILSVRC2012_img_train.tar`. On first use, check the full file against the original archive's MD5, `1d675b47d978889d74fa0da5fadfb00e`:

```bash
md5sum /absolute/path/to/ILSVRC2012_img_train.tar
```

After a successful check, record the date, algorithm and measured digest in `ILSVRC2012_img_train.tar.verified` beside the archive. The scanner requires this verification record. It avoids reading the roughly 138 GiB archive for another full-file check on every scan.

```bash
vision-microscope scan \
  --archive /absolute/path/to/ILSVRC2012_img_train.tar \
  --output vision-repro/outputs/imagenet_scan --batch-size 24 --workers 4
```

The scan selects 50 images per class, for 50,000 training images, and saves the cropped inputs, channel scores, sampling records, heatmaps and browser examples. Both positive and negative examples use this subset. Preserve completed scans for reuse. The [method guide](methodology.md#natural-examples-and-comparability) explains the reference site's different sampling scope.

## 3. Build anchors and generate the five views

```bash
vision-microscope anchors --scan vision-repro/outputs/imagenet_scan \
  --output vision-repro/outputs/all_neurons/anchors

vision-microscope optimize --layer mixed4c --neurons 447 \
  --variant path_positive --steps 1024 \
  --anchors vision-repro/outputs/all_neurons/anchors/mixed4c.pt \
  --output vision-repro/outputs/mixed4c_447_path_positive
```

Use `positive`, `path_positive`, `diversity`, `negative` or `path_negative` as the variant, with a separate output directory for each. Positive and Negative work without anchors. The guided variants use natural-example path anchors; Diversity uses natural scores to scale its loss. Load anchor and resume files from trusted runs, since resume files include Python and random-generator state.

## 4. Generate the full catalog

```bash
vision-microscope campaign init --root vision-repro/outputs/all_neurons --batch-images 8
vision-microscope campaign run --root vision-repro/outputs/all_neurons --workers 1
vision-microscope campaign status --root vision-repro/outputs/all_neurons
```

The catalog requests 29,020 channel–variant pairs. Each Diversity pair produces four images. `--batch-images` must be a positive multiple of four; Diversity uses one quarter as many channels per batch. Start with one worker and measure resource use before increasing concurrency. Independent-channel losses are summed to preserve each target's gradient scale as the batch size changes.

A directory lock permits one supervisor per campaign. It records GPU and host-memory telemetry, reduces concurrency under memory pressure, and manages its own children. Keep protocol-tracked source files fixed during a campaign. Restart the same campaign to continue pending work. Targets without usable anchors are marked `unavailable`.

## 5. Crop examples and export the viewer

Receptive-field crops and aligned heatmaps are prepared on CPU from an existing scan:

```bash
vision-microscope crop-examples \
  --scan vision-repro/outputs/imagenet_scan \
  --model-source source/inceptionv1/lucent/lucent/modelzoo/inceptionv1/InceptionV1.py \
  --output vision-repro/outputs/rf_lossless --workers 2
```

Add `--targets conv2d0:0 mixed3a:0 mixed4c:447 mixed5b:0` to check a small selection first. Each channel has 48 display positions: Top 16, two groups of eight Additional Examples, and the negative Top 16. Crop size follows the channel's branch and is capped at 224 pixels. Boundary crops are shifted inside the input; deep receptive fields can cover the full image.

Optionally create smaller previews while retaining the lossless snapshot:

```bash
python -m vision_repro.compress_rf_site_snapshot \
  vision-repro/outputs/rf_lossless vision-repro/outputs/rf_preview
```

Compression checks each atlas and retains lossless encoding when the quality threshold is missed. Export the selected runs and examples:

```bash
vision-microscope export --runs vision-repro/outputs/all_neurons \
  --sample-run vision-repro/outputs/imagenet_scan \
  --rf-snapshot vision-repro/outputs/rf_preview \
  --output vision-repro/outputs/site
python -m http.server 8766 --directory vision-repro/outputs/site
```

The export includes completed batches and verified display assets. Checkpoints, the original ImageNet archive and local provenance paths stay in the source run. Missing targets have placeholders. Use a fresh export directory. Duplicate channel–variant results across input runs are rejected, so select the intended run explicitly.

For reference comparisons, add `--reference-catalog /path/to/reference-catalog.json` using the [documented format](data-format.md#static-export). It records verified availability and links to official PNGs. Without a supplied catalog, reference images remain placeholders; the original channel page is still linked.

A static server can host the exported directory. Full galleries may contain several GiB of images and should be stored separately from source code. ImageNet and external assets retain their redistribution terms. The optional object-storage service uses `--published-api`; see [Hosting](hosting.md).

## 6. Refit color statistics

The package includes fitted numerical statistics. To fit statistics from your own archive:

```bash
python vision-repro/scripts/fit_imagenet_color.py \
  --archive /absolute/path/to/ILSVRC2012_img_train.tar
```

This writes a timestamped directory under `vision-repro/outputs/color_statistics/`. Pass its `statistics.json` through `optimize --fitted-color`. The supplied A parameters are in `defaults/A.json` within the package. Record changed settings in a new run directory.
