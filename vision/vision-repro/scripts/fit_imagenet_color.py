"""Fit RGB PCA statistics from class-balanced samples inside nested ImageNet tar.

Read-only archive access; no extraction, downloads or changes to dataset bytes.
"""

import argparse
import hashlib
import io
import json
import platform
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import PIL
from PIL import Image

from vision_repro.color_statistics import RGBMoments
from vision_repro.models import PROJECT_ROOT


def center_crop(image):
    image = image.convert("RGB")
    width, height = image.size
    if width <= height:
        size = (256, int(256 * height / width))
    else:
        size = (int(256 * width / height), 256)
    image = image.resize(size, Image.Resampling.BILINEAR)
    left = round((size[0] - 224) / 2)
    top = round((size[1] - 224) / 2)
    return image.crop((left, top, left + 224, top + 224))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--pixels-per-image", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--max-classes", type=int, help="Only for a small pipeline smoke test")
    args = parser.parse_args()
    if args.per_class < 2 or not 1 <= args.pixels_per_image <= 224 * 224:
        parser.error("Need >=2 images/class and 1..50176 pixels/image")
    run = (
        PROJECT_ROOT / "outputs/color_statistics" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    )
    run.mkdir(parents=True, exist_ok=False)
    print(json.dumps({"statistics_dir": str(run)}), flush=True)
    all_moments = RGBMoments()
    halves = [RGBMoments(), RGBMoments()]
    image_count = 0
    dataset_path = args.archive.resolve()
    marker = dataset_path.with_suffix(dataset_path.suffix + ".verified")
    provenance = {
        "archive": str(dataset_path),
        "archive_size": dataset_path.stat().st_size,
        "expected_official_md5": "1d675b47d978889d74fa0da5fadfb00e",
        "verification_marker": str(marker) if marker.exists() else None,
        "verification_marker_text": marker.read_text() if marker.exists() else None,
        "md5_recomputed_this_run": False,
        "seed": args.seed,
        "per_class": args.per_class,
        "pixels_per_image": args.pixels_per_image,
        "preprocessing": "PIL convert RGB; bilinear resize shorter side 256 with integer long side; center crop224; no EXIF transpose; encoded RGB /255; no mean offset applied to optimization",
        "sampling": "uniform without replacement within each class, classes equal weight; pixels uniform without replacement per image; RNG SeedSequence(seed,class index)",
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "python": platform.python_version(),
        "source_sha256": {
            "fit_imagenet_color.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "color_statistics.py": hashlib.sha256(
                (PROJECT_ROOT / "src/vision_repro/color_statistics.py").read_bytes()
            ).hexdigest(),
        },
    }
    (run / "provenance.json").write_text(json.dumps(provenance, indent=2))
    manifest_path = run / "selected_images.jsonl"
    try:
        with tarfile.open(dataset_path, "r:") as outer, manifest_path.open("w") as manifest:
            classes = sorted(
                (m for m in outer.getmembers() if m.isfile() and m.name.endswith(".tar")),
                key=lambda m: m.name,
            )
            if not args.max_classes and len(classes) != 1000:
                raise ValueError(f"Expected 1000 ImageNet class archives, got {len(classes)}")
            if args.max_classes:
                classes = classes[: args.max_classes]
            for index, member in enumerate(classes):
                rng = np.random.default_rng(np.random.SeedSequence([args.seed, index]))
                with (
                    outer.extractfile(member) as class_stream,
                    tarfile.open(fileobj=class_stream, mode="r:") as inner,
                ):
                    images = sorted(
                        (
                            m
                            for m in inner.getmembers()
                            if m.isfile() and m.name.lower().endswith((".jpeg", ".jpg", ".png"))
                        ),
                        key=lambda m: m.name,
                    )
                    picks = rng.choice(len(images), args.per_class, replace=False)
                    for selection_index, pick in enumerate(picks):
                        image_member = images[int(pick)]
                        with inner.extractfile(image_member) as source:
                            payload = source.read()
                        with Image.open(io.BytesIO(payload)) as image:
                            original_mode, original_size = image.mode, image.size
                            crop = np.asarray(center_crop(image))
                        positions = rng.choice(224 * 224, args.pixels_per_image, replace=False)
                        pixels = crop.reshape(-1, 3)[positions].astype(np.float64) / 255
                        all_moments.update(pixels)
                        split = int(selection_index >= args.per_class // 2)
                        halves[split].update(pixels)
                        manifest.write(
                            json.dumps(
                                {
                                    "class_tar": member.name,
                                    "image": image_member.name,
                                    "image_sha256": hashlib.sha256(payload).hexdigest(),
                                    "original_mode": original_mode,
                                    "original_size": original_size,
                                    "pixel_positions_sha256": hashlib.sha256(
                                        positions.astype("<i8").tobytes()
                                    ).hexdigest(),
                                    "stability_split": split,
                                }
                            )
                            + "\n"
                        )
                        image_count += 1
                if (index + 1) % 25 == 0 or index == 0:
                    manifest.flush()
                    print(
                        json.dumps(
                            {
                                "classes_done": index + 1,
                                "classes_total": len(classes),
                                "images_done": image_count,
                            }
                        ),
                        flush=True,
                    )
        stats = {
            **all_moments.result(),
            "image_count": image_count,
            "class_count": len(classes),
            "is_full_imagenet": False,
            "mean_used_in_optimization": False,
            "provenance": provenance,
            "selected_images_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "split_half_statistics": [m.result() for m in halves],
        }
        (run / "statistics.json").write_text(json.dumps(stats, indent=2))
        print(
            json.dumps(
                {
                    "statistics_file": str(run / "statistics.json"),
                    "image_count": image_count,
                    "pixel_count": stats["pixel_count"],
                    "matrix": stats["normalized_matrix"],
                }
            ),
            flush=True,
        )
    except Exception as error:
        (run / "failure.json").write_text(
            json.dumps({"error": str(error), "images_done": image_count})
        )
        raise


if __name__ == "__main__":
    main()
