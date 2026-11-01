# Optimization method

## Model and target

We use Lucent's PyTorch port of the historical TensorFlow Inception5h model with frozen weights. Inputs are 224 × 224 RGB images in `[0,1]`, transformed to `255*x-117` before the model. This checkpoint is distinct from torchvision GoogLeNet and Goodfire's retrained models; channel identities depend on the checkpoint.

The channel objective is a normalized Gaussian-weighted activation:

`a(x) = sum_uv G_sigma(u,v) * F_channel(x)[u,v]`

Spatial coordinates span `[-0.5,0.5]`. The Gaussian is centered at zero, uses `sigma=0.12`, and sums to 1. Positive maximizes post-ReLU activation. Negative minimizes the corresponding branch's signed pre-ReLU activation. Evaluation checks `relu(pre) == post` and records the channel's branch mapping.

## Configuration A: image parameterization

- Initialize independent Gaussian Fourier coefficients with standard deviation 0.01 and default seed 0.
- Scale frequencies by `1/f^1.5` and smoothly increase the low-pass cutoff ratio from 0.15 to 0.45.
- Map decorrelated optimization coordinates into RGB using a fitted covariance factor, `U*sqrt(lambda)`, normalized by its largest column norm. This preserves natural RGB correlations in the rendered image; it does not whiten the final RGB pixels.
- Fit color statistics from 10 ImageNet images per class: 10,000 images and 2,560,000 pixels. This is separate from the 50,000-image retrieval scan.
- Apply a logit chroma gain of 2, then sigmoid to map into `[0,1]`. Anneal Gaussian blur from 2 to 0.5 pixels.
- Optimize with Adam, starting at learning rate 0.03. The annealing schedule lasts 768 steps; subsequent steps retain the final frequency band, blur and learning rate. The default run lasts 1,024 steps.
- Apply random translation, rotation and scale during optimization. Evaluate saved images without these random transforms.

Configuration A uses no alpha compositing, external generative model, display-time color correction, local pixel-similarity loss, logit L2 penalty or total-variation loss.

## Base loss

For each image:

`L = sign*a(T(x)) + 20000*mean_RGB((mean_space(x)-c)^2) + 80000*mean_RGB((mean_periphery(x)-b)^2)`

Here, `T` is the random image transform, `sign=-1` for positive objectives and `sign=+1` for negative objectives. The color targets are `c=(0.65,0.70,0.74)` and `b=(0.68,0.76,0.79)`. The peripheral mean uses smooth radial weights `(1-exp(-r²/(2*0.35²)))²`. These terms constrain regional mean colors rather than individual pixel colors. The cyan-blue appearance partly reflects this explicit prior.

For positive objectives, inactive targets can redirect the ReLU backward gradient while retaining the standard ReLU forward pass. Once all members of a neuron group have positive activation, gradient redirection is disabled for that neuron and is not re-enabled. Negative objectives use pre-ReLU directly without this warm-up. Gradient redirection therefore differs from replacing the positive objective with pre-ReLU activation throughout optimization.

## Five visualization modes

| Mode | Activation objective | Additional term |
| --- | --- | --- |
| Positive | Maximize post-ReLU | None |
| Path-guided + | Maximize post-ReLU | Align with positive natural-example path anchors for the first 64 steps |
| Diversity | Maximize post-ReLU for four images | Penalize Gaussian-weighted Gram cosine similarity in the preceding layer across the four images |
| Negative | Minimize signed pre-ReLU | None |
| Path-guided − | Minimize signed pre-ReLU | Align with negative natural-example path anchors for the first 64 steps |

The path representation combines Gaussian-pooled activation vectors from the preceding catalog layer and the target layer. Set the target channel to zero, normalize each vector, concatenate them, and normalize again. For the first layer, the preceding representation is the RGB input.

Anchors use the top or bottom 60 natural examples from the 50,000-image subset. Pool at each example's activation maximum or minimum, then normalize the mean vector. The guidance term is `1-cos(path,anchor)`, weighted by the median absolute natural-example Gaussian score and applied only during the first 64 steps.

This implementation uses activation vectors as a path proxy, rather than pixel-gradient paths or gradient × activation. The Diversity weight is 0.3 times the natural-example score scale; diversity is enforced within each neuron's four-image group. Tasks without valid natural anchors are marked unavailable.

## Natural examples and activation maps

Scan 50 ImageNet training images per class, for 50,000 images total. Resize the shorter side to 256 pixels and take a 224 × 224 center crop. Rank positive examples by spatial maximum post-ReLU activation and negative examples by spatial minimum signed pre-ReLU activation. These ranking scores differ from the central Gaussian score used for optimization.

Each neuron displays 48 sample positions: Top 16, eight examples near each of two additional percentiles, and the 16 most negative examples. Heatmaps show the native layer response, normalized separately for each image. They are not pixel attribution maps, and their color intensity does not compare absolute activation across images.

The viewer displays optimization outputs without extra sharpening, color grading or background compositing. Runs retain their model, target, data subset and hyperparameters so visual comparisons can be traced to their settings.

## Receptive-field crops

Receptive fields are propagated through the fixed historical InceptionV1 architecture. For a convolution or pooling operation with kernel width `k`, stride `s` and left padding `p`:

```text
r_new = r + (k - 1) * j
j_new = j * s
start_new = start + ((k - 1) / 2 - p) * j
```

Here, `r` is receptive-field width, `j` is the spacing between adjacent feature-cell centers in input coordinates, and `start` is the first cell's center. Input pixel 0 is centered at 0.5. Inception concatenations select the receptive field of each channel's branch. Later convolutions that mix channels use the union of their input dependencies.

For example, `conv2d0:0` has a receptive-field width of 7 pixels. The three convolution branches of `mixed3a` have widths 27, 43 and 59. The width for `mixed4c:447` is 299, capped at 224 for display.

Take a square crop around the activation extremum. At image boundaries, shift the crop inside the image instead of adding black padding. Sample the heatmap using the actual feature-cell centers in the same crop coordinate system, and transform the extremum marker accordingly. Cropping changes the display region; the optimization objective and sample-ranking definitions stay the same.
