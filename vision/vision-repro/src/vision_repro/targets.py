"""One activation definition shared by optimization and later dataset ranking."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NeuronTarget:
    layer: str
    channel: int
    aggregation: str = "center"
    gaussian_sigma: float = 0.2

    def __call__(self, activation):
        if activation.ndim != 4:
            raise ValueError(f"Expected BCHW, received {tuple(activation.shape)}")
        if not 0 <= self.channel < activation.shape[1]:
            raise ValueError(f"Channel {self.channel} outside [0, {activation.shape[1]})")
        field = activation[:, self.channel]
        if self.aggregation == "center":
            return field[:, field.shape[-2] // 2, field.shape[-1] // 2].mean()
        if self.aggregation == "mean":
            return field.mean()
        if self.aggregation == "max":
            return field.flatten(1).max(dim=1).values.mean()
        if self.aggregation == "gaussian":
            if self.gaussian_sigma <= 0:
                raise ValueError("gaussian_sigma must be positive")
            # Coordinates cover [-0.5, 0.5], symmetric even for even-sized maps.
            y = torch.linspace(-0.5, 0.5, field.shape[-2], device=field.device, dtype=field.dtype)
            x = torch.linspace(-0.5, 0.5, field.shape[-1], device=field.device, dtype=field.dtype)
            weights = torch.exp(
                -(y[:, None].square() + x[None, :].square()) / (2 * self.gaussian_sigma**2)
            )
            weights = weights / weights.sum()
            return (field * weights).sum(dim=(-2, -1)).mean()
        raise ValueError(f"Unknown aggregation: {self.aggregation}")


class ActivationCapture:
    def __init__(self, model, layer):
        modules = dict(model.named_modules())
        if layer not in modules:
            raise ValueError(f"Unknown layer {layer!r}; inspect model.named_modules()")
        self.value = None
        self.handle = modules[layer].register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.value = output

    def close(self):
        self.handle.remove()
