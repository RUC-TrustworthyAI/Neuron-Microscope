# Vision visualization method

**English** · [简体中文](methodology_zh.md)

## Model and targets

The implementation freezes a pinned Lucent conversion of TensorFlow Inception5h. Inputs are 224 × 224 RGB values in `[0,1]`, transformed to `255*x-117` before the forward pass. The [reference metadata](https://already-clean.goodfire.com/microscope/inceptionv1/meta.json) also identifies Lucent pretrained weights. This module records its source revision and checkpoint digest; exact weight identity with the reference has not been verified.

The channel objective is a normalized Gaussian-weighted activation:

`a(x) = sum_uv G_sigma(u,v) * F_channel(x)[u,v]`.

Coordinates range over `[-0.5,0.5]`, the Gaussian is centered at zero, `sigma=0.12`, and its weights sum to 1. Positive maximizes post-ReLU activation. Negative minimizes the corresponding branch's signed pre-ReLU activation. Evaluation checks `relu(pre) == post` and records the channel's branch mapping.

## Configuration A

- Independent Gaussian Fourier coefficients start with standard deviation 0.01; the default seed is 0.
- Frequency scaling uses `1/f^1.5`. The low-pass cutoff increases smoothly from 0.15 to 0.45.
- The fitted RGB covariance factor is `U*sqrt(lambda)`, normalized by the largest column norm. Decorrelation in optimization coordinates preserves natural RGB correlations when mapped back to the image.
- Color statistics use 10 images per ImageNet class: 10,000 images and 2,560,000 pixels. This fitting process is separate from the 50,000-image retrieval scan.
- Logit chroma gain is 2, followed by a sigmoid map to `[0,1]`. Gaussian blur decreases from 2 to 0.5 pixels.
- Adam starts at learning rate 0.03. The annealing schedule lasts 768 steps; later steps retain its final frequency band, blur and learning rate. The configured run length is 1,024 steps.
- Optimization applies random translation, rotation and scale changes. Evaluation scores use images without these random transforms.

Configuration A uses the Fourier parameterization and the losses below. Its alpha compositing, neighboring-pixel, logit-L2 and TV options are disabled. Saved previews retain the optimizer's RGB output.

## Base loss

For each image:

`L = sign*a(T(x)) + 20000*mean_RGB((mean_space(x)-c)^2) + 80000*mean_RGB((mean_periphery(x)-b)^2)`.

Positive uses `sign=-1`; negative uses `sign=+1`. The color targets are `c=(0.65,0.70,0.74)` and `b=(0.68,0.76,0.79)`. Peripheral averaging uses the smooth radial weight `(1-exp(-r²/(2*0.35²)))²`. The constraints apply to regional means, so individual peripheral pixels can differ. The explicit prior contributes to the cyan/blue appearance of generated images.

While a positive target is inactive, the optimizer can redirect ReLU backward gradients; the forward pass retains ordinary ReLU. Once all members for a channel have positive activation, redirection is disabled for that channel for the remainder of the run. Negative uses pre-ReLU directly. The positive objective itself remains post-ReLU.

## Five views

| View | Activation objective | Additional term |
| --- | --- | --- |
| Positive | Maximize post-ReLU | Base loss only |
| Path-guided + | Maximize post-ReLU | Positive natural-path alignment for the first 64 steps |
| Diversity | Maximize post-ReLU for four images | Previous-layer Gaussian-weighted Gram-cosine penalty within the four-image group |
| Negative | Minimize signed pre-ReLU | Base loss only |
| Path-guided − | Minimize signed pre-ReLU | Negative natural-path alignment for the first 64 steps |

The path vector concatenates Gaussian-pooled vectors from the previous catalog layer and the current layer, with the target channel zeroed. Each vector is normalized before concatenation, then the concatenation is normalized again. For the first layer, the previous representation is input RGB.

For each pole, anchors use the top/minimum 60 natural examples in the scan. Pooling is centered on each example's response extremum, and the anchor is the unit-normalized mean. The guidance term is `1-cos(path,anchor)`, weighted by the median absolute natural Gaussian score, and applies only during the first 64 steps.

This is a defined activation-vector proxy. Comparing guided results requires the vector definition, weight and guidance schedule. Diversity weight is 0.3 times the natural score scale; comparisons stay within a channel's four members. Targets with unusable anchors are marked `unavailable`.

## Natural examples and comparability

The scan selects 50 images per class, for 50,000 ImageNet training images. It resizes the short side to 256, then center-crops to 224. Positive examples are ranked by spatial maximum post-ReLU activation; negative examples by spatial minimum signed pre-ReLU activation. Optimization uses a central Gaussian score, so its scores and the retrieval scores have separate definitions.

The [reference page](https://already-clean.goodfire.com/microscope/inceptionv1.html) describes positive examples from 50,000 training images and a negative tail over all 1,281,167 training images. This module uses its class-balanced 50,000-image subset for both poles. Natural-example tails therefore depend on both the scan size and the selected images.

Each channel has 48 display positions: Top 16, two groups of eight examples near additional percentiles, and the negative Top 16. Heatmaps display layer-grid responses normalized per image. Their intensity cannot be used to compare absolute activation across images or establish pixel-level attribution.

The displayed optimization images retain their recorded colors and sharpness. The model, objective, data subset and hyperparameters accompany results. Assessing a channel's function requires its measured responses and controls as well as visual inspection.

## Receptive-field display

Receptive fields are computed from the fixed historical InceptionV1 architecture. For a convolution or pool with kernel width `k`, stride `s` and left padding `p`:

```text
r_new = r + (k - 1) * j
j_new = j * s
start_new = start + ((k - 1) / 2 - p) * j
```

The center of input pixel 0 is 0.5. Concatenated Inception layers use the field of the channel's branch; convolutions mixing input channels use the union of their dependency ranges. For example, `conv2d0:0` has width 7, the three convolution branches in `mixed3a` have widths 27, 43 and 59, and `mixed4c:447` has width 299, capped at 224 for display.

Square crops are centered on response extrema and shifted inside the image at boundaries. Heatmaps are sampled using the actual feature-cell centers in the same crop coordinates, and extremum markers are transformed with them. These crops describe architectural dependencies; target scores and example rankings retain their original definitions.
