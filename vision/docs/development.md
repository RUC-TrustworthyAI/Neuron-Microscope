# Development and software checks

Use the dedicated Python 3.11+ environment described in the [vision guide](../README.md#installation-and-usage), with the package's `dev` dependencies installed. Run these checks from the `vision/` directory:

```bash
CUDA_VISIBLE_DEVICES= python -m pytest vision-repro/tests -q
node --check --input-type=module < vision-repro/src/vision_repro/web/microscope/app.js
node vision-repro/tests/test_viewer.mjs
node hosting/test-media.mjs
vision-microscope catalog
```

The CPU suite covers signed targets, independent-channel batches, image parameterization, color statistics, queue coverage, resume, receptive-field geometry, audited seed retries, complete-result merges and static export. Snapshot tests also cover zero-activation inclusion, original Diversity members, repair provenance and reuse of verified cloud objects. The Node checks cover the viewer and optional media API. Fixtures use synthetic data and do not need a model download, ImageNet archive, GPU or paid API.

For numerical optimizer or model-loading changes, also run a short pretrained check on a designated CUDA host after provisioning the [model assets](../source/inceptionv1/README.md):

```bash
vision-microscope doctor
vision-microscope optimize --layer mixed4c --neurons 447 \
  --variant positive --steps 8 --output vision-repro/outputs/smoke
```

Check the saved initial/final scores, target definition, model identity and export. An eight-step execution checks the software path; visual quality and semantic agreement require the full experiment and its evidence.
