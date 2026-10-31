"""Module 1 baseline: reproducible positive/negative activation maximization.

This is not the essay's 2026 recipe or path-guided feature visualization.
"""

import argparse
import csv
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

from .models import DEFAULT_MODEL_DIR, PROJECT_ROOT, load_inceptionv1, preprocess
from .targets import ActivationCapture, NeuronTarget


class ImageParameter(torch.nn.Module):
    def __init__(self, size, mode, device):
        super().__init__()
        self.size, self.mode = size, mode
        if mode == "fft":
            shape = (1, 3, size, size // 2 + 1, 2)
            self.coefficients = torch.nn.Parameter(torch.randn(shape, device=device) * 0.01)
            fy = torch.fft.fftfreq(size, device=device)[:, None]
            fx = torch.fft.rfftfreq(size, device=device)[None, :]
            scale = 1.0 / (fx.square() + fy.square()).sqrt().clamp_min(1 / size)
            self.register_buffer("scale", scale)
        else:
            self.coefficients = torch.nn.Parameter(
                torch.randn(1, 3, size, size, device=device) * 0.01
            )

    def forward(self):
        value = self.coefficients
        if self.mode == "fft":
            spectrum = torch.view_as_complex(value.contiguous()) * self.scale
            value = torch.fft.irfft2(spectrum, s=(self.size, self.size), norm="ortho")
        return value.sigmoid()


def augment(image, config):
    angle = math.radians(random.uniform(-config["rotation"], config["rotation"]))
    scale = random.uniform(config["scale_min"], config["scale_max"])
    j = config["jitter"]
    dx, dy = random.randint(-j, j), random.randint(-j, j)
    # grid_sample maps output coordinates back to input coordinates.
    theta = image.new_tensor(
        [
            [math.cos(angle) / scale, -math.sin(angle) / scale, 2 * dx / image.shape[-1]],
            [math.sin(angle) / scale, math.cos(angle) / scale, 2 * dy / image.shape[-2]],
        ]
    )[None]
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    return F.grid_sample(image, grid, padding_mode="reflection", align_corners=False)


def save_image(image, path):
    array = image.detach()[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    Image.fromarray((array * 255).round().astype(np.uint8)).save(path)


def save_curve(history, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(7, 3.5))
    axes.plot(
        [row["step"] for row in history],
        [row["augmented_activation"] for row in history],
        linewidth=1,
    )
    axes.set(xlabel="Optimization step", ylabel="Activation (randomly transformed input)")
    axes.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/feature_viz/inceptionv1_mixed4e_55.yaml",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/feature_viz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--channel", type=int)
    parser.add_argument("--layer")
    parser.add_argument("--direction", choices=["positive", "negative"])
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    for key in ("steps", "seed", "channel", "layer", "direction"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if config["steps"] < 1 or config["image_size"] < 32 or config["save_every"] < 1:
        parser.error("steps/save_every must be positive and image_size >= 32")
    if config["direction"] == "negative" and not config["layer"].endswith("_pre_relu_conv"):
        parser.error(
            "Negative-pole visualization requires an explicitly selected pre-ReLU conv node; post-ReLU mixed blocks cannot go negative"
        )
    if config["parameterization"] not in ("fft", "pixel"):
        parser.error("parameterization must be fft or pixel")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing an accidental CPU fallback")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    model, manifest = load_inceptionv1(args.model_dir, args.device)
    target = NeuronTarget(config["layer"], config["channel"], config["aggregation"])
    capture = ActivationCapture(model, target.layer)
    parameter = ImageParameter(config["image_size"], config["parameterization"], args.device)
    optimizer = torch.optim.Adam(parameter.parameters(), lr=config["learning_rate"])
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run = (
        args.output_root
        / f"{stamp}_{target.layer}_ch{target.channel}_{config['direction']}_seed{config['seed']}"
    )
    run.mkdir(parents=True, exist_ok=False)
    yaml_text = yaml.safe_dump(config, sort_keys=False)
    (run / "config.yaml").write_text(yaml_text)
    metadata = {
        "model": manifest,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "method": "baseline_fft_or_pixel_affine_robustness",
        "path_guided": False,
        "redirected_relu": False,
        "input_preprocessing": "RGB [0,1] * 255 - 117",
        "bitwise_reproducibility_guaranteed": False,
    }
    (run / "metadata.json").write_text(json.dumps(metadata, indent=2))
    sign = 1 if config["direction"] == "positive" else -1
    history = []
    try:
        with torch.no_grad():
            model(preprocess(parameter()))
            initial = float(target(capture.value))
        save_image(parameter(), run / "initial.png")
        for step in range(1, config["steps"] + 1):
            optimizer.zero_grad(set_to_none=True)
            image = parameter()
            model(preprocess(augment(image, config)))
            activation = target(capture.value)
            tv = (image[..., 1:, :] - image[..., :-1, :]).square().mean()
            tv = tv + (image[..., :, 1:] - image[..., :, :-1]).square().mean()
            loss = -sign * activation + config["tv_weight"] * tv
            loss = loss + config["l2_weight"] * (image - 0.5).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite objective at step {step}")
            loss.backward()
            grad = parameter.coefficients.grad
            if grad is None or not torch.isfinite(grad).all():
                raise RuntimeError(f"Invalid image gradient at step {step}")
            grad_norm = float(grad.norm())
            optimizer.step()
            history.append(
                {
                    "step": step,
                    "augmented_activation": float(activation.detach()),
                    "loss": float(loss.detach()),
                    "gradient_norm": grad_norm,
                }
            )
            if step == 1 or step % config["save_every"] == 0 or step == config["steps"]:
                save_image(parameter(), run / f"step_{step:04d}.png")
                print(json.dumps(history[-1]), flush=True)
        with torch.no_grad():
            final_image = parameter()
            model(preprocess(final_image))
            final = float(target(capture.value))
        save_image(final_image, run / "optimized.png")
        with (run / "history.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        save_curve(history, run / "activation_curve.png")
        summary = {
            "initial_unaugmented_activation": initial,
            "final_unaugmented_activation": final,
            "signed_improvement": sign * (final - initial),
            "output_dir": str(run),
        }
        (run / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    finally:
        capture.close()


if __name__ == "__main__":
    main()
