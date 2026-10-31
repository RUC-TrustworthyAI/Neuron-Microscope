"""CPU-only receptive-field geometry for the historical Lucent InceptionV1.

Coordinates are pixel-edge coordinates: input pixel zero has centre 0.5.
The radius is an architectural dependency bound, not a gradient attribution.
Mixed-layer channels use their own branch kernel and the union of all incoming
channels. No guessed parameters from the Goodfire export are used here.
"""
from dataclasses import dataclass
import ast
import math
from pathlib import Path

BRANCHES = ("1x1", "3x3", "5x5", "pool_reduce")
MIXED = ("mixed3a", "mixed3b", "mixed4a", "mixed4b", "mixed4c",
         "mixed4d", "mixed4e", "mixed5a", "mixed5b")


@dataclass(frozen=True)
class Field:
    size: int = 1
    jump: int = 1
    start: float = 0.5

    def operation(self, kernel, stride=1, pad_left=0):
        return Field(self.size + (kernel-1)*self.jump, self.jump*stride,
                     self.start + ((kernel-1)/2-pad_left)*self.jump)

    def bounds(self, y, x):
        cy, cx = self.start+y*self.jump, self.start+x*self.jump
        return (cx-self.size/2, cy-self.size/2, cx+self.size/2, cy+self.size/2)


def model_geometry(source):
    """Read widths from the model AST without importing torch or allocating a model."""
    widths = {}
    for node in ast.walk(ast.parse(Path(source).read_text())):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr.endswith("_pre_relu_conv"):
                keywords = {k.arg: k.value for k in node.value.keywords}
                if "out_channels" in keywords:
                    widths[target.attr] = ast.literal_eval(keywords["out_channels"])
    results = {}
    f = Field().operation(7, 2, 2)
    results["conv2d0"] = [f] * widths["conv2d0_pre_relu_conv"]
    f = f.operation(3, 2, 0)  # asymmetric pad on right/bottom only
    results["conv2d1"] = [f] * widths["conv2d1_pre_relu_conv"]
    f = f.operation(3, 1, 1)
    results["conv2d2"] = [f] * widths["conv2d2_pre_relu_conv"]
    f = f.operation(3, 2, 0)
    for layer in MIXED:
        if layer in ("mixed4a", "mixed5a"):
            f = f.operation(3, 2, 0)
        fields = [f, f.operation(3, 1, 1), f.operation(5, 1, 2), f.operation(3, 1, 1)]
        results[layer] = [field for branch, field in zip(BRANCHES, fields)
                          for _ in range(widths[f"{layer}_{branch}_pre_relu_conv"])]
        f = max(fields, key=lambda item: item.size)
    return results


def crop_box(field, location, image_size=224, mode="fit", scale=1., minimum=0):
    """Return requested and actually sampled boxes; never hide the policy difference.

    intersect: true RF intersected with valid pixels, potentially rectangular.
    fit: cap side to image size, shift the square inside the input at boundaries.
    """
    y, x = location
    requested = field.bounds(y, x)
    if mode == "intersect":
        box = (max(0, math.floor(requested[0])), max(0, math.floor(requested[1])),
               min(image_size, math.ceil(requested[2])), min(image_size, math.ceil(requested[3])))
    elif mode == "fit":
        side = min(image_size, max(1, minimum, int(math.ceil(field.size * scale))))
        cy, cx = field.start+y*field.jump, field.start+x*field.jump
        x0 = min(max(int(math.floor(cx-side/2)), 0), image_size-side)
        y0 = min(max(int(math.floor(cy-side/2)), 0), image_size-side)
        box = (x0, y0, x0+side, y0+side)
    else:
        raise ValueError(mode)
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError((field, location, box))
    return requested, box


def sampled_heatmap(raw, field, box, pixels=112):
    """Bilinear map sampling aligned to exact feature-cell centres, not guessed /grid."""
    import numpy as np
    x0, y0, x1, y1 = box
    x = (x0+(np.arange(pixels)+.5)*(x1-x0)/pixels-field.start)/field.jump
    y = (y0+(np.arange(pixels)+.5)*(y1-y0)/pixels-field.start)/field.jump
    x = np.clip(x, 0, raw.shape[1]-1); y = np.clip(y, 0, raw.shape[0]-1)
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    jx, jy = np.minimum(ix+1, raw.shape[1]-1), np.minimum(iy+1, raw.shape[0]-1)
    wx, wy = x-ix, y-iy
    return ((1-wy[:,None])*((1-wx)*raw[iy[:,None],ix]+wx*raw[iy[:,None],jx])+
            wy[:,None]*((1-wx)*raw[jy[:,None],ix]+wx*raw[jy[:,None],jx]))
