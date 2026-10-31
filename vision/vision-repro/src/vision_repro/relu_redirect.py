"""PyTorch port of Lucid's conditional, descent-sign ReLU gradient override.

Rule source (Lucid Authors, Apache-2.0):
https://github.com/tensorflow/lucid/blob/master/lucid/misc/redirected_relu_grad.py
Forward values/weights are unchanged. This is a surrogate backward, not the
derivative of the post-ReLU objective. Call with a MINIMIZED loss, not a score.
"""

from dataclasses import dataclass

import torch


@dataclass
class RedirectState:
    global_step: int = 0
    last_global_step: int = 16
    record: bool = False
    opened_batch_ops: int = 0
    passed_negative_elements: int = 0

    def reset_counts(self):
        self.opened_batch_ops = 0
        self.passed_negative_elements = 0


class _Redirect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, state):
        ctx.save_for_backward(value)
        ctx.state = state
        ctx.active = state.global_step <= state.last_global_step
        return value.clamp_min(0)

    @staticmethod
    def backward(ctx, grad):
        (value,) = ctx.saved_tensors
        # Match the source's x < 0 test, including its derivative choice at zero.
        normal = torch.where(value < 0, 0.0, grad)
        if not ctx.active:
            return normal, None
        if value.ndim < 2:
            raise ValueError("Lucid rule requires a batch dimension and rank >=2")
        blocked = normal.flatten(1).norm(dim=1) == 0
        mask = blocked.reshape((-1,) + (1,) * (value.ndim - 1))
        redirected = torch.where((value < 0) & (grad > 0), 0.0, grad)
        result = torch.where(mask, redirected, normal)
        if ctx.state.record:
            opened = mask & (value < 0) & (grad < 0)
            ctx.state.opened_batch_ops += int(opened.flatten(1).any(dim=1).sum())
            ctx.state.passed_negative_elements += int(opened.sum())
        return result, None


class LucidRedirectReLU(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def forward(self, value):
        return _Redirect.apply(value, self.state)


def install_redirected_relu(model, state, extra_relu_types=()):
    """Replace verified ReLU wrappers on this model only, before capture hooks."""
    replaced = []
    for name, child in list(model.named_children()):
        if isinstance(child, (torch.nn.ReLU, *extra_relu_types)):
            setattr(model, name, LucidRedirectReLU(state))
            replaced.append(name)
        else:
            replaced.extend(
                f"{name}.{x}"
                for x in install_redirected_relu(child, state, extra_relu_types)
            )
    return replaced
