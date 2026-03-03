#!/usr/bin/env python3
import os
import glob
import shutil
from pathlib import Path
import argparse

import cv2
import numpy as np


# -----------------------------
# Image discovery
# -----------------------------
IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

def list_images(folder: str):
    files = []
    p = Path(folder)
    for e in IMAGE_EXTS:
        files.extend(glob.glob(str(p / "**" / e), recursive=True))
    return sorted(set(files))


# -----------------------------
# Dedup helpers
# -----------------------------
def load_thumb_gray(path: Path, size=(64, 64)):
    """
    Returns a small grayscale thumbnail as uint8 array (H,W).
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    thumb = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return thumb

def a_hash(thumb_gray: np.ndarray):
    """
    Average hash (64-bit if thumb is 8x8, but we're using 64x64 and then downsampling to 8x8).
    Returns a Python int bitstring.
    """
    small = cv2.resize(thumb_gray, (8, 8), interpolation=cv2.INTER_AREA)
    avg = small.mean()
    bits = (small >= avg).astype(np.uint8).ravel()

    # pack bits into int
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h

def hamming_distance_int(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def diff_score_L1(a: np.ndarray, b: np.ndarray) -> int:
    """
    L1 difference over the thumbnail pixels.
    """
    # use int16 to avoid uint8 wrap
    return int(np.abs(a.astype(np.int16) - b.astype(np.int16)).sum())


def unique_path(out_dir: Path, desired_name: str) -> Path:
    """
    If desired_name exists, append _0001, _0002, ...
    """
    base = Path(desired_name).stem
    ext = Path(desired_name).suffix.lower()
    candidate = out_dir / f"{base}{ext}"
    if not candidate.exists():
        return candidate

    k = 1
    while True:
        candidate = out_dir / f"{base}_{k:04d}{ext}"
        if not candidate.exists():
            return candidate
        k += 1


# -----------------------------
# Main consolidation
# -----------------------------
def consolidate(
    input_dirs,
    output_dir,
    move=False,
    thumb_size=(64, 64),
    diff_threshold=100,
    max_hamming=0,
):
    """
    Strategy:
      1) For each image, compute (aHash, thumb).
      2) Compare only against prior images with "close enough" aHash:
           - if max_hamming == 0: only exact hash matches are compared
           - else: compare if hamming(aHash, existingHash) <= max_hamming
      3) If any candidate has L1 thumb diff < diff_threshold => duplicate.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_images = []
    for d in input_dirs:
        all_images.extend(list_images(d))

    if not all_images:
        raise SystemExit("No images found in input dirs.")

    # Map hash -> list of (thumb, output_path)
    # If you set max_hamming > 0, we'll still use this map but also scan nearby hashes.
    hash_buckets = {}

    copied = 0
    skipped = 0
    unreadable = 0

    def iter_candidate_thumbs(cur_hash: int):
        if max_hamming == 0:
            for item in hash_buckets.get(cur_hash, []):
                yield item
            return

        # With hamming search, scan all hashes (OK for small sets).
        # If this gets big (10k+), we can implement LSH / BK-tree later.
        for h, items in hash_buckets.items():
            if hamming_distance_int(cur_hash, h) <= max_hamming:
                for item in items:
                    yield item

    for i, img_path_str in enumerate(all_images, 1):
        img_path = Path(img_path_str)

        thumb = load_thumb_gray(img_path, size=thumb_size)
        if thumb is None:
            print(f"[WARN] Unreadable: {img_path}")
            unreadable += 1
            continue

        h = a_hash(thumb)

        is_dup = False
        for existing_thumb, existing_out in iter_candidate_thumbs(h):
            score = diff_score_L1(thumb, existing_thumb)
            if score < diff_threshold:
                is_dup = True
                break

        if is_dup:
            skipped += 1
            continue

        # Not a duplicate => copy/move into output_dir
        dest = unique_path(out_dir, img_path.name)
        if move:
            shutil.move(str(img_path), str(dest))
        else:
            shutil.copy2(str(img_path), str(dest))

        hash_buckets.setdefault(h, []).append((thumb, dest))
        copied += 1

        if i % 200 == 0:
            print(f"[INFO] Processed {i}/{len(all_images)} | copied={copied} skipped={skipped} unreadable={unreadable}")

    print("\nDone.")
    print(f"Total found:   {len(all_images)}")
    print(f"Copied/moved: {copied}")
    print(f"Skipped dups: {skipped}")
    print(f"Unreadable:   {unreadable}")
    print(f"Output dir:   {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Consolidate images from folders into one folder with content-based dedup.")
    ap.add_argument("--input", nargs="+", required=True, help="One or more input directories.")
    ap.add_argument("--output", required=True, help="Output directory.")
    ap.add_argument("--move", action="store_true", help="Move files instead of copy.")
    ap.add_argument("--thumb", default="64", help="Thumbnail size (e.g. 64 or 128).")
    ap.add_argument("--diff_threshold", type=int, default=100, help="Duplicate threshold for L1 thumbnail diff.")
    ap.add_argument("--max_hamming", type=int, default=0, help="Allow near hashes (0 = exact hash only).")
    args = ap.parse_args()

    ts = int(args.thumb)
    consolidate(
        input_dirs=args.input,
        output_dir=args.output,
        move=args.move,
        thumb_size=(ts, ts),
        diff_threshold=args.diff_threshold,
        max_hamming=args.max_hamming,
    )


if __name__ == "__main__":
    main()