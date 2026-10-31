"""Load a locally provisioned historical InceptionV1, with no implicit downloads."""

import hashlib
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = Path(os.environ.get("VNI_MODEL_DIR", PROJECT_ROOT.parent / "source" / "inceptionv1"))


def load_inceptionv1(model_dir=DEFAULT_MODEL_DIR, device="cuda"):
    model_dir = Path(model_dir).resolve()
    checkout = model_dir / "lucent"
    weights = model_dir / "weights" / "inception5h.pth"
    manifest = model_dir / "manifest.json"
    if not checkout.is_dir() or not weights.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"Model not provisioned completely under {model_dir}")
    metadata = json.loads(manifest.read_text())
    with weights.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != metadata["weights_sha256"]:
        raise ValueError("InceptionV1 weight checksum does not match manifest")
    if str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))
    from lucent.modelzoo.inceptionv1 import InceptionV1

    # Lucent defaults to a redirected (modified) ReLU gradient; disable explicitly.
    model = InceptionV1(pretrained=False, redirected_ReLU=False)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.eval().requires_grad_(False)
    return model.to(device), metadata


def preprocess(image):
    """RGB BCHW float [0, 1] -> historical TensorFlow Inception5h input range."""
    return image * 255.0 - 117.0
