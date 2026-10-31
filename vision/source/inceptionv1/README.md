# InceptionV1 assets

This directory contains provenance, not model weights or vendored code.

```text
source/inceptionv1/
  manifest.json
  lucent/                   external checkout, ignored by Git
  weights/inception5h.pth    external weights, ignored by Git
```

Provision [Lucent](https://github.com/greentfrapp/lucent) at commit `2ccc434d80097e58f53cb2a70c60f72902fe93af`. Obtain `inception5h.pth` from the `weights_url` in [manifest.json](manifest.json), subject to the upstream terms. The exact expected SHA-256 is:

```text
44f1f1e0734fa77edd0ab948c0d4d5c9ab6f4efce10b02d793fd14ab2d71a73b
```

Existing assets can be reused through an absolute `VNI_MODEL_DIR`; the directory must have the layout above. No command silently downloads weights. Run `vision-microscope doctor` before optimization: it checks the weight hash, source commit and tracked modifications. Loading verifies the weight hash again.

The model uses 1008 output classes and historical `RGB*255-117` preprocessing. The feature-visualization pipeline stops at the requested intermediate layer and uses 224-pixel input geometry. The [AlreadyClean reference metadata](https://already-clean.goodfire.com/microscope/inceptionv1/meta.json) identifies Lucent pretrained weights; its exact checkpoint digest has not been verified against this manifest.
