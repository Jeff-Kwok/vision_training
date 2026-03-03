#!/usr/bin/env python3
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple

# -------------------
# EDIT THESE
# -------------------
IMAGES_DIR = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/raw/photos/raw_consolidated/standardized_1120_labeled/images"
LABELS_DIR = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/raw/photos/raw_consolidated/standardized_1120_labeled/labels"
OUT_DIR = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/training_sets/dataset_12"

SPLIT = (0.65, 0.25, 0.10)      # train, val, test
SEED = 1337
COPY_MODE = "copy"              # "copy" or "move"
REQUIRE_LABEL = True            # True = skip images missing labels

# Label Studio export metadata files live beside images/labels typically:
# dataset_root/
#   images/
#   labels/
#   classes.txt
#   notes.json
LABEL_STUDIO_ROOT = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/label_studio/dataset_sz50"

# Ultralytics expects "val:" key; some people name the folder "valid".
# Set VAL_KEY_NAME to "val" if your folder is OUT_DIR/val
# Set VAL_KEY_NAME to "valid" if your folder is OUT_DIR/valid
VAL_KEY_NAME = "valid"
# -------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(images_dir: Path) -> List[Path]:
    return sorted([p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def ensure_dirs(out_root: Path, val_folder_name: str):
    for split in ("train", val_folder_name, "test"):
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)


def copy_or_move(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "move":
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def compute_counts(n: int, split: Tuple[float, float, float]) -> Tuple[int, int, int]:
    train = int(n * split[0])
    val = int(n * split[1])
    test = n - train - val
    return train, val, test


def copy_metadata_files(label_studio_root: Path, out_root: Path):
    """Copy classes.txt and notes.json into OUT_DIR if they exist."""
    copied = []
    for name in ("classes.txt", "notes.json"):
        src = label_studio_root / name
        if src.exists():
            dst = out_root / name
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def load_categories_from_notes(notes_path: Path):
    """Return list of (id:int, name:str) sorted by id."""
    data = json.loads(notes_path.read_text())
    cats = data.get("categories", [])
    parsed = []
    for c in cats:
        if "id" in c and "name" in c:
            parsed.append((int(c["id"]), str(c["name"])))
    parsed.sort(key=lambda x: x[0])
    return parsed


def write_data_yaml(out_root: Path, categories: List[Tuple[int, str]], val_folder_name: str):
    """
    Create OUT_DIR/data.yaml:
      path: <OUT_DIR>
      train: train
      val: <val_folder_name>
      test: test
      nc: <N>
      names:
        0: bolt
        1: shackle
    """
    nc = len(categories)
    lines = []
    lines.append(f"path: {out_root}")
    lines.append("train: train")
    lines.append(f"val: {val_folder_name}")
    lines.append("test: test")
    lines.append("")
    lines.append(f"nc: {nc}")
    lines.append("names:")
    for cid, name in categories:
        lines.append(f"  {cid}: {name.strip().lower()}")
    (out_root / "data.yaml").write_text("\n".join(lines) + "\n")


def main():
    images_dir = Path(IMAGES_DIR)
    labels_dir = Path(LABELS_DIR)
    out_root = Path(OUT_DIR)
    ls_root = Path(LABEL_STUDIO_ROOT)

    assert abs(sum(SPLIT) - 1.0) < 1e-6, f"SPLIT must sum to 1.0, got {SPLIT}"
    if COPY_MODE not in ("copy", "move"):
        raise SystemExit("COPY_MODE must be 'copy' or 'move'")
    if VAL_KEY_NAME not in ("val", "valid"):
        raise SystemExit("VAL_KEY_NAME should be 'val' or 'valid' (or set to your folder name).")

    imgs = list_images(images_dir)
    if not imgs:
        raise SystemExit(f"No images found in {images_dir}")

    # Pair images with labels
    pairs = []
    missing_labels = []
    for img in imgs:
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            missing_labels.append(img)
            if not REQUIRE_LABEL:
                pairs.append((img, None))

    random.seed(SEED)
    random.shuffle(pairs)

    n = len(pairs)
    n_train, n_val, n_test = compute_counts(n, SPLIT)

    # Create output dirs
    ensure_dirs(out_root, VAL_KEY_NAME)

    # Split lists
    train_items = pairs[:n_train]
    val_items = pairs[n_train:n_train + n_val]
    test_items = pairs[n_train + n_val:]

    splits = (
        ("train", train_items),
        (VAL_KEY_NAME, val_items),
        ("test", test_items),
    )

    moved_or_copied = 0
    labels_written = 0

    for split_name, items in splits:
        out_img_dir = out_root / split_name / "images"
        out_lbl_dir = out_root / split_name / "labels"

        for img, lbl in items:
            img_dst = out_img_dir / img.name
            copy_or_move(img, img_dst, COPY_MODE)
            moved_or_copied += 1

            if lbl is not None:
                lbl_dst = out_lbl_dir / f"{img.stem}.txt"
                copy_or_move(lbl, lbl_dst, COPY_MODE)
                labels_written += 1

    # Copy metadata files
    copied_meta = copy_metadata_files(ls_root, out_root)

    # Build data.yaml from notes.json
    notes_path = out_root / "notes.json"
    if notes_path.exists():
        categories = load_categories_from_notes(notes_path)
        if not categories:
            print("[WARN] notes.json exists but no categories were found; data.yaml not written.")
        else:
            write_data_yaml(out_root, categories, VAL_KEY_NAME)
            print(f"[OK] Wrote data.yaml -> {out_root / 'data.yaml'}")
    else:
        print("[WARN] notes.json not found in OUT_DIR; data.yaml not written.")

    print("\n=== Split complete ===")
    print(f"Images source:   {images_dir}")
    print(f"Labels source:   {labels_dir}")
    print(f"LabelStudio root:{ls_root}")
    print(f"Output dataset:  {out_root}")
    print(f"Mode:            {COPY_MODE}")
    print(f"Seed:            {SEED}")
    print(f"Val folder name: {VAL_KEY_NAME}")
    print("")
    print(f"Total images considered: {len(imgs)}")
    print(f"Pairs used for split:    {n}")
    print(f"Missing labels:          {len(missing_labels)} (REQUIRE_LABEL={REQUIRE_LABEL})")
    print("")
    print(f"Train: {n_train}  Val: {n_val}  Test: {n_test}")
    print(f"Images written: {moved_or_copied}")
    print(f"Labels written: {labels_written}")

    if copied_meta:
        print("\nMetadata copied:")
        for p in copied_meta:
            print(f"  - {p}")

    if missing_labels:
        print("\nExamples missing labels:")
        for p in missing_labels[:10]:
            print(f"  - {p.name}")


if __name__ == "__main__":
    main()
