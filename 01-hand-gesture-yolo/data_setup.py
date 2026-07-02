"""
Prepare a HaGRID v2 512px subset for YOLOv8 hand-gesture detection.

What it does:
1. Loads HaGRID JSON annotations for 5 target classes.
2. Deterministically samples N images per class using numpy random.
3. Splits sampled images into train/val/test by user_id to avoid leakage.
4. Extracts only the sampled images from the large zip archive.
5. Converts bounding boxes from HaGRID format to YOLO format.
6. Writes YOLO .txt labels and a config.yaml.
"""

# %% Imports
import argparse
import json
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

import numpy as np

# %%-------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Map HaGRID class names -> our YOLO class IDs
CLASS_MAP = {
    "palm": 0,
    "fist": 1,
    "like": 2,
    "peace": 3,
    "no_gesture": 4,
}

# Human-readable names for config.yaml
CLASS_NAMES = {
    0: "open_palm",
    1: "closed_fist",
    2: "thumbs_up",
    3: "peace_sign",
    4: "no_gesture",
}

# Train / val / test ratios. Must sum to 1.0.
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


# %%-------------------------------------------------------------------------
# TypedDict definitions for HaGRID annotation format
# ---------------------------------------------------------------------------


class HaGRIDMeta(TypedDict):
    """Auto-annotated metadata for a single image."""

    age: list[float]
    gender: list[str]
    race: list[str]


class HaGRIDAnnotation(TypedDict):
    """One HaGRID annotation entry keyed by image ID."""

    bboxes: list[list[float]]
    user_id: str
    labels: list[str]
    united_bbox: list[list[float]] | None
    united_label: list[str] | None
    meta: HaGRIDMeta
    hand_landmarks: list[list[list[float]]]


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------


def load_annotations(
    annotations_dir: Path, hagrid_classes: list[str]
) -> dict[str, dict[str, HaGRIDAnnotation]]:
    """
    Load annotations for target classes across train/val/test.

    Returns:
        records: {hagrid_class: {image_id: annotation_dict}}
    """
    records: dict[str, dict[str, HaGRIDAnnotation]] = {
        cls: {} for cls in hagrid_classes
    }

    for split in ["train", "val", "test"]:
        split_dir = annotations_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Annotation split not found: {split_dir}")

        for cls in hagrid_classes:
            path = split_dir / f"{cls}.json"
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for image_id, annotation in data.items():
                records[cls][image_id] = annotation

    return records


# ---------------------------------------------------------------------------
# Sampling and splitting
# ---------------------------------------------------------------------------


def sample_image_ids(
    records: dict[str, dict[str, HaGRIDAnnotation]],
    samples_per_class: int,
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    """
    Deterministically sample image IDs for each class.

    If a class has fewer images than requested, use all of them.
    """
    sampled = {}
    for cls, annotations in records.items():
        all_ids = list(annotations.keys())
        n_available = len(all_ids)
        n_sample = min(samples_per_class, n_available)

        sampled[cls] = rng.choice(all_ids, size=n_sample, replace=False).tolist()
        print(f"  {cls}: sampled {n_sample:,} / {n_available:,} available")

    return sampled


def split_by_user(
    sampled_ids: dict[str, list[str]],
    records: dict[str, dict[str, HaGRIDAnnotation]],
    split_ratios: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, dict[str, list[str]]]:
    """
    Split sampled IDs into train/val/test by user_id.

    Splitting by user avoids data leakage: the same person does not appear
    in multiple splits.
    """
    splits = {cls: {"train": [], "val": [], "test": []} for cls in sampled_ids}

    for cls, ids in sampled_ids.items():
        # Group image IDs by the person who appears in them
        user_to_ids = defaultdict(list)
        for image_id in ids:
            user_id = records[cls][image_id].get("user_id", image_id)
            user_to_ids[user_id].append(image_id)

        users = list(user_to_ids.keys())
        rng.shuffle(users)

        n_users = len(users)
        n_train = int(n_users * split_ratios["train"])
        n_val = int(n_users * split_ratios["val"])

        train_users = users[:n_train]
        val_users = users[n_train : n_train + n_val]
        test_users = users[n_train + n_val :]

        for user in train_users:
            splits[cls]["train"].extend(user_to_ids[user])
        for user in val_users:
            splits[cls]["val"].extend(user_to_ids[user])
        for user in test_users:
            splits[cls]["test"].extend(user_to_ids[user])

    return splits


# ---------------------------------------------------------------------------
# YOLO format conversion
# ---------------------------------------------------------------------------


def hagrid_to_yolo_bbox(
    x_min: float, y_min: float, width: float, height: float
) -> tuple[float, float, float, float]:
    """
    HaGRID: [x_min, y_min, width, height]
    YOLO:   [x_center, y_center, width, height]
    """
    x_center = x_min + width / 2.0
    y_center = y_min + height / 2.0
    return x_center, y_center, width, height


def write_yolo_label(image_id: str, record: HaGRIDAnnotation, label_path: Path) -> None:
    """
    Write a YOLO .txt label file for one image.

    Keeps all bounding boxes that belong to our 5 target classes.
    This correctly handles images with multiple hands.
    """
    lines = []
    for bbox, label in zip(record["bboxes"], record["labels"]):
        if label not in CLASS_MAP:
            continue

        class_id = CLASS_MAP[label]
        x_min, y_min, width, height = bbox
        x_c, y_c, w, h = hagrid_to_yolo_bbox(x_min, y_min, width, height)
        lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    if lines:
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Zip extraction
# ---------------------------------------------------------------------------


def build_zip_member_name(image_id: str, hagrid_class: str, zip_prefix: str) -> str:
    """Guess the full path of an image inside the HaGRID zip."""
    return f"{zip_prefix}/{hagrid_class}/{image_id}.jpg"


def extract_images(
    zip_path: Path,
    splits: dict[str, dict[str, list[str]]],
    records: dict[str, dict[str, HaGRIDAnnotation]],
    output_dir: Path,
    zip_prefix: str,
) -> None:
    """
    Extract only the sampled images from the zip and write YOLO labels.
    """
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Image archive not found: {zip_path}\nPlease download it first."
        )

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = set(zf.namelist())

        for cls, split_dict in splits.items():
            for split, image_ids in split_dict.items():
                img_out = images_dir / split
                lbl_out = labels_dir / split

                for image_id in image_ids:
                    member = build_zip_member_name(image_id, cls, zip_prefix)

                    if member not in namelist:
                        print(f"    Warning: not found in zip: {member}")
                        continue

                    # Read image bytes from zip and write to flat output folder
                    img_bytes = zf.read(member)
                    out_img_path = img_out / f"{image_id}.jpg"
                    out_img_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_img_path, "wb") as f:
                        f.write(img_bytes)

                    # Write corresponding YOLO label
                    record = records[cls][image_id]
                    out_lbl_path = lbl_out / f"{image_id}.txt"
                    write_yolo_label(image_id, record, out_lbl_path)


# ---------------------------------------------------------------------------
# Config file
# ---------------------------------------------------------------------------


def write_config(output_dir: Path, config_path: Path) -> None:
    """Write the YOLO dataset config.yaml."""
    # Use a path relative to the config file so the project stays portable.
    config_parent = config_path.parent
    if config_parent == Path("."):
        config_parent = Path.cwd()
    rel_path = output_dir.resolve().relative_to(config_parent.resolve())

    lines = [
        f"path: {rel_path}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for i in range(len(CLASS_NAMES)):
        lines.append(f"  {i}: {CLASS_NAMES[i]}")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare HaGRID 512px subset for YOLO hand-gesture detection."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/hagrid/raw/annotations"),
        help="Directory containing extracted HaGRID annotations.",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path("data/hagrid/raw/hagridv2_512.zip"),
        help="Path to the HaGRID 512px zip archive.",
    )
    parser.add_argument(
        "--zip-prefix",
        type=str,
        default="HaGRIDv2_dataset_512",
        help="Top-level folder name inside the zip (e.g., HaGRIDv2_dataset_512)."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/hagrid/processed"),
        help="Output directory for processed dataset.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to write YOLO config file.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=1000,
        help="Number of images to sample per class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("HaGRID -> YOLO dataset setup")
    print("=" * 60)
    print(f"Random seed: {args.seed}")
    print(f"Samples per class: {args.samples_per_class}")
    print(f"Output directory: {args.out}")
    print()

    # Deterministic random number generator
    rng = np.random.default_rng(args.seed)

    hagrid_classes = list(CLASS_MAP.keys())

    print("Step 1/5: Loading annotations...")
    records = load_annotations(args.annotations, hagrid_classes)

    print("Step 2/5: Sampling images per class...")
    sampled_ids = sample_image_ids(records, args.samples_per_class, rng)

    print("Step 3/5: Splitting by user_id...")
    splits = split_by_user(sampled_ids, records, SPLIT_RATIOS, rng)

    print("Split sizes:")
    for cls, split_dict in splits.items():
        sizes = {k: len(v) for k, v in split_dict.items()}
        total = sum(sizes.values())
        print(
            f"  {cls:12s}: train={sizes['train']:4d}, val={sizes['val']:4d}, "
            f"test={sizes['test']:4d}, total={total:4d}"
        )
    print()

    # Clean output directory before writing
    if args.out.exists():
        print(f"Step 4/5: Removing existing output directory: {args.out}")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    print("Step 5/5: Extracting images and writing labels...")
    extract_images(args.zip, splits, records, args.out, args.zip_prefix)

    print(f"Writing {args.config}...")
    write_config(args.out, args.config)

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
