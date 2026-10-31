"""Public-description-inspired ordinary positive feature visualization.

Numerical schedules are our choices, not recovered Goodfire hyperparameters.
No path guidance or implicit initialization fallback is used. Optional explicit
pre-ReLU warmup is separately recorded and is not a recovered author recipe.
"""

import argparse
import csv
import hashlib
import json
import math
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from .color_statistics import load_fitted_color
from .models import PROJECT_ROOT, load_inceptionv1, preprocess
from .optimize import augment, save_curve, save_image
from .relu_redirect import RedirectState, install_redirected_relu
from .targets import ActivationCapture, NeuronTarget


def schedule(progress, config):
    """Cosine unlock to full radial support; linear blur anneal to final sigma."""
    fraction = min(max(progress / config["unlock_until"], 0.0), 1.0)
    eased = (1 - math.cos(math.pi * fraction)) / 2
    cutoff = config["initial_band"] + (1 - config["initial_band"]) * eased
    hold = config.get("blur_hold_until", 0.0)
    fraction = min(max((progress - hold) / (config["blur_until"] - hold), 0.0), 1.0)
    sigma = config["blur_start"] + (config["blur_end"] - config["blur_start"]) * fraction
    return cutoff, sigma


def learning_rate_at(progress, config):
    start = config["lr_decay_start"]
    fraction = min(max((progress - start) / (1 - start), 0.0), 1.0)
    factor = (
        config["lr_final_ratio"]
        + (1 - config["lr_final_ratio"]) * (1 + math.cos(math.pi * fraction)) / 2
    )
    return config["learning_rate"] * factor


def optimization_progress(step, config):
    """Allow longer training without stretching the original annealing schedule."""
    return min((step - 1) / max(config.get("schedule_steps", config["steps"]) - 1, 1), 1.0)


def color_penalties(image, logits, config):
    """Mean penalties on actual blurred RGB before augmentation and effective logits."""
    common = logits.mean(dim=1).square().mean()
    luma = (image * image.new_tensor([0.2126, 0.7152, 0.0722])[None, :, None, None]).sum(1)
    brightness = (luma - config.get("luma_target", 0.65)).square().mean()
    weighted = (config.get("common_logit_l2_weight", 0.0) * common
                + config.get("luma_l2_weight", 0.0) * brightness)
    return weighted, common, brightness


def gaussian_blur(image, sigma):
    if sigma <= 1e-6:
        return image
    radius = math.ceil(3 * sigma)
    if radius >= min(image.shape[-2:]):
        raise ValueError("Blur kernel too large for reflection padding")
    axis = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-axis.square() / (2 * sigma**2))
    kernel /= kernel.sum()
    channels = image.shape[1]
    value = F.conv2d(
        F.pad(image, (radius, radius, 0, 0), mode="reflect"),
        kernel[None, None, None, :].expand(channels, 1, 1, -1),
        groups=channels,
    )
    return F.conv2d(
        F.pad(value, (0, 0, radius, radius), mode="reflect"),
        kernel[None, None, :, None].expand(channels, 1, -1, 1),
        groups=channels,
    )


class ScheduledFFT(torch.nn.Module):
    def __init__(self, size, std, device, color_mode="rgb", fft_divisor=1.0, color_matrix=None,
                 pixel_gain=1.0, pixel_bias=0.0, chroma_gain=1.0):
        super().__init__()
        self.size = size
        if not math.isfinite(pixel_gain) or pixel_gain <= 0 or not math.isfinite(pixel_bias):
            raise ValueError("Pixel gain must be positive and pixel mapping parameters finite")
        self.pixel_gain, self.pixel_bias = pixel_gain, pixel_bias
        if not math.isfinite(chroma_gain) or chroma_gain <= 0:
            raise ValueError("chroma_gain must be finite and positive")
        if chroma_gain != 1 and color_mode != "lucid_svd":
            raise ValueError("chroma_gain currently requires lucid_svd")
        if not math.isfinite(fft_divisor) or fft_divisor <= 0:
            raise ValueError("fft_divisor must be finite and positive")
        self.fft_divisor = fft_divisor
        if color_mode not in {"rgb", "lucid_svd", "fitted"}:
            raise ValueError("Unknown color mode")
        # Lucid Authors, Apache-2.0: lucid/optvis/param/color.py.
        # Empirical ImageNet color covariance square-root factors, normalized by
        # the largest column norm exactly as in Lucid. Not fitted to our images.
        color = torch.tensor(
            [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]], device=device
        )
        color = color / torch.linalg.vector_norm(color, dim=0).max()
        if color_mode == "rgb":
            color = torch.eye(3, device=device)
        if color_mode == "fitted":
            color = torch.as_tensor(color_matrix, dtype=torch.float32, device=device)
            if (color.shape != (3, 3) or not torch.isfinite(color).all()
                    or abs(float(torch.linalg.det(color))) < 1e-8):
                raise ValueError("Expected finite invertible fitted color matrix")
        color[:, 1:] *= chroma_gain
        self.register_buffer("color_matrix", color)
        base_noise = torch.randn(1, 3, size, size // 2 + 1, 2, device=device)
        self.base_noise_sha256 = hashlib.sha256(
            base_noise.detach().cpu().numpy().tobytes()
        ).hexdigest()
        self.coefficients = torch.nn.Parameter(base_noise * std)
        fy = torch.fft.fftfreq(size, device=device)[:, None]
        fx = torch.fft.rfftfreq(size, device=device)[None, :]
        radius = (fx.square() + fy.square()).sqrt()
        self.register_buffer("scale", radius.clamp_min(1 / size).reciprocal())
        self.register_buffer("radius", radius / radius.max())

    def initialize_rgb(self, image):
        if tuple(image.shape) != (1, 3, self.size, self.size):
            raise ValueError("Initialization must be exact-sized RGB BCHW")
        with torch.no_grad():
            logits = (torch.logit(image.clamp(0.001, 0.999)) - self.pixel_bias) / self.pixel_gain
            # Invert the color map so both modes start at the SAME RGB image.
            logits = (
                torch.linalg.solve(self.color_matrix, logits.movedim(1, -1).unsqueeze(-1))
                .squeeze(-1)
                .movedim(-1, 1)
            )
            spectrum = torch.fft.rfft2(logits, norm="ortho") / self.scale * self.fft_divisor
            self.coefficients.copy_(torch.view_as_real(spectrum))

    def band_mask(self, cutoff):
        if cutoff >= 1:
            return torch.ones_like(self.radius)
        # Raised-cosine taper over the final 20% of the admitted radius.
        phase = ((self.radius - cutoff * 0.8) / (cutoff * 0.2)).clamp(0, 1)
        return (1 + torch.cos(math.pi * phase)) / 2

    def rgb_logits(self, cutoff=1.0):
        spectrum = (
            torch.view_as_complex(self.coefficients.contiguous())
            * self.scale
            * self.band_mask(cutoff)
        )
        latent = (
            torch.fft.irfft2(spectrum, s=(self.size, self.size), norm="ortho") / self.fft_divisor
        )
        logits = torch.einsum("ij,bjhw->bihw", self.color_matrix, latent)
        return self.pixel_gain * logits + self.pixel_bias

    def forward(self, cutoff=1.0, sigma=0.0, return_logits=False):
        logits = self.rgb_logits(cutoff)
        image = logits.sigmoid()
        # Blur the actual RGB fed to the model, not just its saved preview.
        image = gaussian_blur(image, sigma)
        return (image, logits) if return_logits else image


def validate(config):
    for key in ["common_logit_l2_weight", "luma_l2_weight"]:
        if not math.isfinite(config.get(key, 0.0)) or config.get(key, 0.0) < 0:
            raise ValueError(f"Invalid {key}")
    if not math.isfinite(config.get("luma_target", 0.65)) or not 0 <= config.get("luma_target", 0.65) <= 1:
        raise ValueError("luma_target must be in [0,1]")
    if not 0 <= config.get("blur_hold_until", 0.0) < config.get("blur_until", 1.0):
        raise ValueError("blur_hold_until must be before blur_until")
    if (not math.isfinite(config.get("chroma_gain", 1.0))
            or config.get("chroma_gain", 1.0) <= 0
            or (config.get("chroma_gain", 1.0) != 1 and config["color_mode"] != "lucid_svd")):
        raise ValueError("Invalid chroma_gain or color mode")
    if config.get("schedule_steps", config["steps"]) < 1:
        raise ValueError("schedule_steps must be positive")
    if (not math.isfinite(config.get("pixel_gain", 1.0))
            or config.get("pixel_gain", 1.0) <= 0
            or not math.isfinite(config.get("pixel_bias", 0.0))):
        raise ValueError("Invalid pixel mapping gain or bias")
    redirect = config.get("relu_gradient_mode", "standard")
    if redirect not in {"standard", "lucid_redirect"}:
        raise ValueError("Unknown ReLU gradient mode")
    redirect_steps = config.get("relu_redirect_steps", 17)
    if type(redirect_steps) is not int or redirect_steps < 1:
        raise ValueError("relu_redirect_steps must be a positive integer")
    if redirect != "standard" and config["pre_relu_warmup_steps"]:
        raise ValueError("Do not combine redirected ReLU with pre-ReLU warmup")
    if not math.isfinite(config["fft_divisor"]) or config["fft_divisor"] <= 0:
        raise ValueError("Invalid FFT divisor")
    if not 0 < config["lr_final_ratio"] <= 1 or not 0 <= config["lr_decay_start"] < 1:
        raise ValueError("Invalid learning-rate schedule")
    warmup = config["pre_relu_warmup_steps"]
    if not 0 <= warmup < config["steps"]:
        raise ValueError("Warmup must be nonnegative and shorter than total steps")
    if warmup and (config["layer"], config["channel"]) != ("mixed4c", 447):
        raise ValueError(
            "Warmup mapping verified only for mixed4c:447 -> mixed4c_5x5_pre_relu_conv:63"
        )
    if not math.isfinite(config["logit_l2_weight"]) or config["logit_l2_weight"] < 0:
        raise ValueError("logit_l2_weight must be finite and nonnegative")
    if config["direction"] != "positive" or config["aggregation"] != "gaussian":
        raise ValueError("This entry point requires positive Gaussian aggregation")
    if config["color_mode"] not in {"rgb", "lucid_svd", "fitted"}:
        raise ValueError("Unknown color mode")
    if config["color_mode"] == "fitted" and not config.get("color_statistics_path"):
        raise ValueError("Fitted color mode requires a recorded statistics file")
    if config["color_mode"] != "fitted" and config.get("color_statistics_path"):
        raise ValueError("Statistics file only allowed in fitted color mode")
    if config["layer"].endswith("_pre_relu_conv"):
        raise ValueError("This entry point requires a post-ReLU layer")
    if config["steps"] < 1 or config["save_every"] < 1 or config["image_size"] != 224:
        raise ValueError("Positive steps/save_every and 224px model input required")
    if (
        not 0 < config["initial_band"] <= 1
        or not 0 < config["unlock_until"] <= 1
        or not 0 < config["blur_until"] <= 1
    ):
        raise ValueError("Invalid frequency/blur schedule")
    if (
        min(config["blur_start"], config["blur_end"]) < 0
        or config["init_attempts"] < 1
        or config["init_std"] <= 0
        or not math.isfinite(config["init_std"])
        or config["gaussian_sigma"] <= 0
    ):
        raise ValueError("Invalid initialization or Gaussian settings")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/feature_viz/inceptionv1_mixed4c447_positive.yaml",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--init-std", type=float)
    parser.add_argument("--fft-divisor", type=float)
    parser.add_argument("--lr-final-ratio", type=float)
    parser.add_argument("--lr-decay-start", type=float)
    parser.add_argument("--blur-end", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--color-mode", choices=["rgb", "lucid_svd", "fitted"])
    parser.add_argument("--color-statistics-path", type=Path)
    parser.add_argument("--logit-l2-weight", type=float)
    parser.add_argument("--pre-relu-warmup-steps", type=int)
    parser.add_argument("--relu-gradient-mode", choices=["standard", "lucid_redirect"])
    parser.add_argument("--relu-redirect-steps", type=int)
    parser.add_argument(
        "--init-image", type=Path, help="Optional synthetic warm start; recorded, never implicit"
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    config.setdefault("color_mode", "rgb")
    config.setdefault("logit_l2_weight", 0.0)
    config.setdefault("common_logit_l2_weight", 0.0)
    config.setdefault("luma_l2_weight", 0.0)
    config.setdefault("luma_target", 0.65)
    config.setdefault("fft_divisor", 1.0)
    config.setdefault("pixel_gain", 1.0)
    config.setdefault("pixel_bias", 0.0)
    config.setdefault("chroma_gain", 1.0)
    config.setdefault("blur_hold_until", 0.0)
    config.setdefault("lr_final_ratio", 1.0)
    config.setdefault("lr_decay_start", 0.5)
    config.setdefault("pre_relu_warmup_steps", 0)
    config.setdefault("relu_gradient_mode", "standard")
    # Preserve the earlier literal Lucid 0..16 boundary unless explicitly set.
    config.setdefault("relu_redirect_steps", 17)
    if args.relu_redirect_steps is not None:
        config["relu_redirect_steps"] = args.relu_redirect_steps
    if args.relu_gradient_mode is not None:
        config["relu_gradient_mode"] = args.relu_gradient_mode
    if args.pre_relu_warmup_steps is not None:
        config["pre_relu_warmup_steps"] = args.pre_relu_warmup_steps
    if config["pre_relu_warmup_steps"] and args.init_image is not None:
        parser.error("This warmup experiment requires random FFT initialization, not an image")
    if args.logit_l2_weight is not None:
        config["logit_l2_weight"] = args.logit_l2_weight
    if args.color_mode is not None:
        config["color_mode"] = args.color_mode
    if args.color_statistics_path is not None:
        config["color_statistics_path"] = str(args.color_statistics_path)
    for name in [
        "steps",
        "seed",
        "init_std",
        "fft_divisor",
        "lr_final_ratio",
        "lr_decay_start",
        "blur_end",
    ]:
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    validate(config)
    fitted_matrix, fitted_provenance = None, None
    if config["color_mode"] == "fitted":
        statistics_path = Path(config["color_statistics_path"])
        if not statistics_path.is_absolute():
            statistics_path = PROJECT_ROOT / statistics_path
        fitted_matrix, fitted_provenance = load_fitted_color(statistics_path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no silent CPU fallback")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    model, manifest = load_inceptionv1(device=args.device)
    redirect_state = (
        RedirectState(last_global_step=config["relu_redirect_steps"] - 1)
        if config["relu_gradient_mode"] == "lucid_redirect"
        else None
    )
    redirected_modules = []
    if redirect_state:
        # Historical Lucent wraps F.relu in ReluLayer rather than nn.ReLU.
        # Verified wrapper only; do not enable Lucent's different, leaky backward.
        from lucent.modelzoo.inceptionv1.helper_layers import ReluLayer

        redirected_modules = install_redirected_relu(model, redirect_state, (ReluLayer,))
    if redirect_state and not redirected_modules:
        raise RuntimeError("No verified ReLU modules found; override not installed")
    target = NeuronTarget(config["layer"], config["channel"], "gaussian", config["gaussian_sigma"])
    capture = ActivationCapture(model, target.layer)
    warmup_steps = config["pre_relu_warmup_steps"]
    pre_target = (
        NeuronTarget("mixed4c_5x5_pre_relu_conv", 63, "gaussian", config["gaussian_sigma"])
        if warmup_steps
        else None
    )
    pre_capture = ActivationCapture(model, pre_target.layer) if pre_target is not None else None
    run = (
        PROJECT_ROOT
        / "outputs/positive_recipe"
        / (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
            + f"_{target.layer}_ch{target.channel}_seed{config['seed']}"
        )
    )
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    metadata = {
        "model": manifest,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "method": "ordinary_positive_gaussian_fft_band_unlock_blur_v5",
        "fft_divisor": config["fft_divisor"],
        "learning_rate_schedule": {
            "initial": config["learning_rate"],
            "decay_start_fraction": config["lr_decay_start"],
            "final_ratio": config["lr_final_ratio"],
            "type": "cosine",
        },
        "explicit_pre_relu_warmup": {
            "steps": warmup_steps,
            "pre_target": "mixed4c_5x5_pre_relu_conv:63" if warmup_steps else None,
            "schedule": "global total-step schedule; no reset",
            "optimizer": "same Adam instance and state throughout",
            "transition_check": "scheduled unaugmented and 4 affine samples must have post-ReLU >0",
        },
        "pixel_mapping": {"formula": "sigmoid(pixel_gain * RGB_latent + pixel_bias)",
                          "gain": config["pixel_gain"], "bias": config["pixel_bias"]},
        "color_penalty_definitions": {
            "common_logit_l2": "mean over pixels and batch of square(mean RGB effective logits); no sum factor of 9",
            "luma_l2": "mean((encoded RGB luma-target)^2), actual blurred RGB before augmentation; weights .2126,.7152,.0722",
            "note": "Luma target is our explicit image prior, not recovered author method",
        },
        "logit_l2_definition": "mean(square(effective RGB logits after color map, active band mask, pixel gain and bias, before sigmoid/blur/affine)); no threshold",
        "color_mode": config["color_mode"],
        "fitted_color_statistics": fitted_provenance,
        "color_matrix_source": "https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/color.py"
        if config["color_mode"] == "lucid_svd"
        else None,
        "path_guided": False,
        "redirected_relu": bool(redirect_state),
        "relu_gradient_rule": {
            "mode": config["relu_gradient_mode"],
            "source": "https://github.com/tensorflow/lucid/blob/master/lucid/misc/redirected_relu_grad.py",
            "active_global_steps_inclusive": [0, redirect_state.last_global_step]
            if redirect_state else [],
            "enabled_update_count": config["relu_redirect_steps"] if redirect_state else 0,
            "note": "Descent-sign surrogate, per-op per-batch gating. Lucid literal boundary is 0..16 (17 updates); explicit update-count option can shorten this.",
            "replaced_modules": redirected_modules,
        },
        "pre_relu_fallback": False,
        "hyperparameters": "our choices; not exact recovered author recipe",
        "preprocessing": "RGB [0,1] * 255 - 117",
        "target": {
            "layer": target.layer,
            "channel": target.channel,
            "nonlinearity": "post-ReLU",
            "aggregation": "unit-sum Gaussian over feature-map coordinates [-0.5,0.5]",
            "sigma": target.gaussian_sigma,
        },
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ["positive.py", "targets.py", "optimize.py", "models.py",
                         "relu_redirect.py", "color_statistics.py"]
        },
    }
    (run / "metadata.json").write_text(json.dumps(metadata, indent=2))
    initialization_image = None
    if args.init_image is not None:
        args.init_image = args.init_image.resolve()
        with Image.open(args.init_image) as input_image:
            initialization_image = (
                torch.from_numpy(np.asarray(input_image.convert("RGB")).copy())
                .permute(2, 0, 1)[None]
                .to(args.device)
                .float()
                / 255
            )
        metadata["initialization"] = {
            "kind": "explicit_synthetic_warm_start",
            "path": str(args.init_image),
            "sha256": hashlib.sha256(args.init_image.read_bytes()).hexdigest(),
            "warning": "Not a from-noise reproduction; source was generated by earlier pre-ReLU baseline",
        }
    else:
        metadata["initialization"] = {
            "kind": "random_fft",
            "selection": "first nonzero target-gradient candidate; all attempts recorded",
        }
    (run / "metadata.json").write_text(json.dumps(metadata, indent=2))
    attempts, history = [], []
    try:
        cutoff, sigma = schedule(0, config)
        for attempt in range(config["init_attempts"]):
            parameter = ScheduledFFT(
                config["image_size"],
                config["init_std"],
                args.device,
                config["color_mode"],
                config["fft_divisor"],
                color_matrix=fitted_matrix,
                pixel_gain=config["pixel_gain"],
                pixel_bias=config["pixel_bias"],
                chroma_gain=config["chroma_gain"],
            )
            if initialization_image is not None:
                parameter.initialize_rgb(initialization_image)
            model(preprocess(parameter(cutoff, sigma)))
            score = pre_target(pre_capture.value) if warmup_steps else target(capture.value)
            # Redirected gradients depend on the LOSS sign. Positive-score
            # differentiation would incorrectly block the desired ascent path.
            gradient = torch.autograd.grad(-score, parameter.coefficients)[0]
            norm = float(gradient.norm())
            attempts.append(
                {
                    "attempt": attempt,
                    "activation": float(score.detach()),
                    "target_gradient_norm": norm,
                    "base_fft_noise_sha256": parameter.base_noise_sha256,
                }
            )
            if torch.isfinite(gradient).all() and norm > 1e-8:
                break
            if initialization_image is not None:
                raise RuntimeError("Explicit synthetic initialization has no valid target gradient")
        else:
            raise RuntimeError(
                "All random initializations inactive: refusing to silently optimize a dead post-ReLU or substitute pre-ReLU"
            )
        (run / "initialization.json").write_text(json.dumps(attempts, indent=2))
        save_image(parameter(cutoff, sigma), run / "initial_scheduled.png")
        # Fixed evaluation: full bands + final blur, identical before/after.
        with torch.no_grad():
            initial_image = parameter(1, config["blur_end"])
            model(preprocess(initial_image))
            initial = float(target(capture.value))
        save_image(initial_image, run / "initial.png")
        optimizer = torch.optim.Adam(parameter.parameters(), lr=config["learning_rate"])
        for step in range(1, config["steps"] + 1):
            if redirect_state:
                redirect_state.global_step = step - 1
                redirect_state.record = False
                redirect_state.reset_counts()
            progress = optimization_progress(step, config)
            cutoff, sigma = schedule(progress, config)
            current_lr = learning_rate_at(progress, config)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            optimizer.zero_grad(set_to_none=True)
            image, logits = parameter(cutoff, sigma, return_logits=True)
            model(preprocess(augment(image, config)))
            post_activation = target(capture.value)
            pre_activation = pre_target(pre_capture.value) if warmup_steps else None
            stage = "pre_relu_warmup" if step <= warmup_steps else "post_relu"
            if redirect_state and redirect_state.global_step <= redirect_state.last_global_step:
                stage = "post_relu_redirect"
            activation = pre_activation if stage == "pre_relu_warmup" else post_activation
            neuron_grad = torch.autograd.grad(
                -activation, parameter.coefficients, retain_graph=True
            )[0]
            norm = float(neuron_grad.norm())
            if not torch.isfinite(neuron_grad).all() or norm <= 1e-8:
                raise RuntimeError(
                    f"Inactive/invalid target gradient at step {step}; run marked failed"
                )
            tv = (image[..., 1:, :] - image[..., :-1, :]).square().mean() + (
                image[..., :, 1:] - image[..., :, :-1]
            ).square().mean()
            logit_l2 = logits.square().mean()
            weighted_logit_l2 = config["logit_l2_weight"] * logit_l2
            extra_penalty, common_logit_l2, luma_l2 = color_penalties(image, logits, config)
            color_regularizer = weighted_logit_l2 + extra_penalty
            regularizer_grad = torch.autograd.grad(
                color_regularizer, parameter.coefficients, retain_graph=True
            )[0]
            regularizer_norm = float(regularizer_grad.norm())
            loss = -activation + config["tv_weight"] * tv + color_regularizer
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite objective")
            if redirect_state:
                redirect_state.record = True
            loss.backward()
            if redirect_state:
                redirect_state.record = False
            if not torch.isfinite(parameter.coefficients.grad).all():
                raise RuntimeError("Non-finite total gradient")
            optimizer.step()
            history.append(
                {
                    "step": step,
                    "stage": stage,
                    "redirect_opened_batch_ops": redirect_state.opened_batch_ops if redirect_state else 0,
                    "redirect_passed_negative_elements": redirect_state.passed_negative_elements if redirect_state else 0,
                    "augmented_activation": float(activation.detach()),
                    "post_relu_activation": float(post_activation.detach()),
                    "pre_relu_activation": float(pre_activation.detach())
                    if pre_activation is not None
                    else None,
                    "loss": float(loss.detach()),
                    "target_gradient_norm": norm,
                    "logit_l2": float(logit_l2.detach()),
                    "weighted_logit_l2": float(weighted_logit_l2.detach()),
                    "common_logit_l2": float(common_logit_l2.detach()),
                    "luma_l2": float(luma_l2.detach()),
                    "weighted_color_regularizer": float(color_regularizer.detach()),
                    "regularizer_gradient_norm": regularizer_norm,
                    "band_cutoff": cutoff,
                    "blur_sigma_pixels": sigma,
                    "learning_rate": current_lr,
                }
            )
            if step == 1 or step % config["save_every"] == 0 or step == config["steps"]:
                save_image(parameter(cutoff, sigma), run / f"step_{step:04d}.png")
                if config.get("save_fixed_checkpoints", False):
                    with torch.no_grad():
                        fixed_image = parameter(1, config["blur_end"])
                        model(preprocess(fixed_image))
                        fixed_score = float(target(capture.value))
                        save_image(fixed_image, run / f"eval_{step:04d}.png")
                        torch.save(parameter.state_dict(), run / f"parameter_{step:04d}.pt")
                        (run / f"eval_{step:04d}.json").write_text(json.dumps({
                            "step": step, "activation": fixed_score,
                            "definition": "unaugmented, full bands, final blur, configured target",
                        }))
                print(json.dumps(history[-1]), flush=True)
            if step == warmup_steps:
                # Keep exactly the same parameter, Adam state and global schedules.
                rng_state = random.getstate()
                checks = []
                try:
                    with torch.no_grad():
                        image = parameter(cutoff, sigma)
                        for check in range(5):
                            model(preprocess(image if check == 0 else augment(image, config)))
                            checks.append(float(target(capture.value)))
                finally:
                    random.setstate(rng_state)
                (run / "transition.json").write_text(
                    json.dumps(
                        {
                            "after_step": step,
                            "post_relu_scores_unaugmented_then_4_augmented": checks,
                            "adam_step": float(optimizer.state[parameter.coefficients]["step"]),
                            "passed": all(value > 0 for value in checks),
                        },
                        indent=2,
                    )
                )
                save_image(image, run / "warmup_end.png")
                if not all(value > 0 for value in checks):
                    raise RuntimeError(
                        "Fixed warmup did not robustly activate post-ReLU; refusing a silent continuation"
                    )
        with torch.no_grad():
            final_image = parameter(1, config["blur_end"])
            model(preprocess(final_image))
            final = float(target(capture.value))
            final_logit_l2 = float(parameter.rgb_logits(1).square().mean())
            unblurred_image = parameter(1, 0)
            model(preprocess(unblurred_image))
            final_unblurred = float(target(capture.value))
        save_image(final_image, run / "optimized.png")
        save_image(unblurred_image, run / "optimized_unblurred.png")
        torch.save(parameter.state_dict(), run / "parameter.pt")
        torch.save(final_image.detach().cpu(), run / "optimized_tensor.pt")
        summary = {
            "initial_unaugmented_activation": initial,
            "final_unaugmented_activation": final,
            "final_unblurred_activation": final_unblurred,
            "signed_improvement": final - initial,
            "final_logit_l2": final_logit_l2,
            "final_weighted_logit_l2": config["logit_l2_weight"] * final_logit_l2,
            "pre_relu_warmup_steps": warmup_steps,
            "relu_redirect_enabled_updates": min(config["relu_redirect_steps"], config["steps"])
            if redirect_state else 0,
            "post_relu_steps": config["steps"] - warmup_steps,
            "optimizer_total_steps": float(optimizer.state[parameter.coefficients]["step"]),
            "evaluation": "unaugmented, full bands, final blur; same Gaussian target before/after",
            "output_dir": str(run),
        }
        (run / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    except Exception as error:
        (run / "failure.json").write_text(
            json.dumps({"error": str(error), "initialization_attempts": attempts})
        )
        raise
    finally:
        if history:
            with (run / "history.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            save_curve(history, run / "activation_curve.png")
        capture.close()
        if pre_capture is not None:
            pre_capture.close()


if __name__ == "__main__":
    main()
