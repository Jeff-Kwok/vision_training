#!/usr/bin/env python3
"""
Partition a Label Studio COCO *segmentation* export into train/val/test folders
and write split COCO JSONs for RF-DETR (or any COCO-seg training).

Expected input layout:
  INPUT_ROOT/
    images/
      *.jpg|*.png|...
    result_coco.json   (or annotations.json etc.)

Output layout:
  OUT_ROOT/
    train/
      images/
      annotations.json
    valid/ (or val/)
      images/
      annotations.json
    test/
      images/
      annotations.json

Notes:
- Keeps COCO ids stable (no reindex needed). Filters images/annotations to split.
- Optionally drops images with no annotations (often desirable for training).
"""

import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple

# -------------------
# EDIT THESE
# -------------------
INPUT_ROOT = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/label_studio/project-20-at-2026-02-25-01-06-9fd59340"      # contains images/ + result_coco.json
COCO_JSON_NAME = "result_coco.json"            # name of your COCO export json file
OUT_ROOT = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/training_sets/dataset_10" # where to write split dataset

SPLIT = (0.70, 0.20, 0.10)   # train, val, test
SEED = 1337
COPY_MODE = "copy"           # "copy" or "move"
VAL_FOLDER_NAME = "valid"    # "val" or "valid" (or anything)
DROP_IMAGES_WITHOUT_ANN = True  # True = only keep images that have >=1 annotation
# -------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def ensure_out_dirs(out_root: Path, val_folder: str):
    for split in ("train", val_folder, "test"):
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)


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


def load_coco(path: Path) -> Dict:
    data = json.loads(path.read_text())
    # basic sanity
    for key in ("images", "annotations"):
        if key not in data:
            raise SystemExit(f"COCO JSON missing key '{key}': {path}")
    # categories might exist; keep them if present
    return data


def index_images_by_id(images: List[Dict]) -> Dict[int, Dict]:
    out = {}
    for im in images:
        out[int(im["id"])] = im
    return out


def group_annotations_by_image(annotations: List[Dict]) -> Dict[int, List[Dict]]:
    m: Dict[int, List[Dict]] = {}
    for ann in annotations:
        iid = int(ann["image_id"])
        m.setdefault(iid, []).append(ann)
    return m


def find_image_file(images_dir: Path, file_name: str) -> Path:
    p = images_dir / file_name
    if p.exists():
        return p
    # fallback: search by stem if extensions differ
    stem = Path(file_name).stem
    matches = [x for x in images_dir.rglob("*") if x.is_file() and x.suffix.lower() in IMAGE_EXTS and x.stem == stem]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SystemExit(f"Multiple matches for image stem '{stem}' under {images_dir}")
    raise SystemExit(f"Image file not found: {file_name} (looked in {images_dir})")


def write_split_coco(out_json: Path, coco: Dict, image_ids: Set[int]):
    images = [im for im in coco["images"] if int(im["id"]) in image_ids]
    anns = [ann for ann in coco["annotations"] if int(ann["image_id"]) in image_ids]

    out = dict(coco)  # shallow copy
    out["images"] = images
    out["annotations"] = anns

    out_json.write_text(json.dumps(out, indent=2))
    return len(images), len(anns)


def main():
    in_root = Path(INPUT_ROOT)
    images_dir = in_root / "images"
    coco_path = in_root / COCO_JSON_NAME
    out_root = Path(OUT_ROOT)

    if COPY_MODE not in ("copy", "move"):
        raise SystemExit("COPY_MODE must be 'copy' or 'move'")
    if abs(sum(SPLIT) - 1.0) > 1e-6:
        raise SystemExit(f"SPLIT must sum to 1.0, got {SPLIT}")

    if not images_dir.exists():
        raise SystemExit(f"Missing images directory: {images_dir}")
    if not coco_path.exists():
        raise SystemExit(f"Missing COCO json: {coco_path}")

    coco = load_coco(coco_path)
    images = coco["images"]
    annotations = coco["annotations"]

    img_by_id = index_images_by_id(images)
    anns_by_img = group_annotations_by_image(annotations)

    # Choose which image ids are eligible for splitting
    all_image_ids = sorted(img_by_id.keys())
    if DROP_IMAGES_WITHOUT_ANN:
        eligible = [iid for iid in all_image_ids if iid in anns_by_img and len(anns_by_img[iid]) > 0]
    else:
        eligible = all_image_ids

    if not eligible:
        raise SystemExit("No eligible images found (check DROP_IMAGES_WITHOUT_ANN / annotations).")

    random.seed(SEED)
    random.shuffle(eligible)

    n = len(eligible)
    n_train, n_val, n_test = compute_counts(n, SPLIT)

    train_ids = set(eligible[:n_train])
    val_ids = set(eligible[n_train:n_train + n_val])
    test_ids = set(eligible[n_train + n_val:])

    ensure_out_dirs(out_root, VAL_FOLDER_NAME)

    # Copy images
    def copy_split(split_name: str, ids: Set[int]):
        out_img_dir = out_root / split_name / "images"
        count = 0
        missing = 0
        for iid in sorted(ids):
            file_name = img_by_id[iid]["file_name"]
            try:
                src = find_image_file(images_dir, file_name)
            except SystemExit:
                missing += 1
                continue
            dst = out_img_dir / src.name
            copy_or_move(src, dst, COPY_MODE)
            count += 1
        return count, missing

    train_copied, train_missing = copy_split("train", train_ids)
    val_copied, val_missing = copy_split(VAL_FOLDER_NAME, val_ids)
    test_copied, test_missing = copy_split("test", test_ids)

    # Write split COCO JSONs
    train_imgs, train_anns = write_split_coco(out_root / "train" / "annotations.json", coco, train_ids)
    val_imgs, val_anns = write_split_coco(out_root / VAL_FOLDER_NAME / "annotations.json", coco, val_ids)
    test_imgs, test_anns = write_split_coco(out_root / "test" / "annotations.json", coco, test_ids)

    print("\n=== COCO split complete ===")
    print(f"Input root:        {in_root}")
    print(f"COCO json:         {coco_path}")
    print(f"Images dir:        {images_dir}")
    print(f"Output root:       {out_root}")
    print(f"Mode:              {COPY_MODE}")
    print(f"Seed:              {SEED}")
    print(f"Val folder:        {VAL_FOLDER_NAME}")
    print(f"Drop empty images: {DROP_IMAGES_WITHOUT_ANN}")
    print("")
    print(f"Eligible images:   {n}")
    print(f"Train/Val/Test:    {len(train_ids)} / {len(val_ids)} / {len(test_ids)}")
    print("")
    print(f"Images copied:     train={train_copied}, {VAL_FOLDER_NAME}={val_copied}, test={test_copied}")
    if (train_missing + val_missing + test_missing) > 0:
        print(f"[WARN] Missing image files not copied: train={train_missing}, {VAL_FOLDER_NAME}={val_missing}, test={test_missing}")
    print("")
    print(f"COCO written:")
    print(f"  train: images={train_imgs}, anns={train_anns} -> {out_root / 'train' / 'annotations.json'}")
    print(f"  {VAL_FOLDER_NAME}: images={val_imgs}, anns={val_anns} -> {out_root / VAL_FOLDER_NAME / 'annotations.json'}")
    print(f"  test:  images={test_imgs}, anns={test_anns} -> {out_root / 'test' / 'annotations.json'}")


if __name__ == "__main__":
    main()