#!/usr/bin/env python3
import argparse
import glob
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

def list_images(folder: Path):
    files = []
    for e in IMAGE_EXTS:
        files.extend(folder.rglob(e))
    # ignore the duplicates folder if it exists
    return sorted([p for p in files if "duplicates" not in p.parts])


def load_bgr(path: Path):
    # IMREAD_COLOR handles most formats; HEIC won't work without extra libs.
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def normalize(img_bgr: np.ndarray, size: int) -> np.ndarray:
    """
    Normalize to grayscale + fixed size.
    This makes "pixel comparison" meaningful across different resolutions.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return small


def diff_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mean absolute pixel difference (0..255).
    Lower = more similar.
    """
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def best_rotation_match_score(candidate_norm: np.ndarray, reference_norm: np.ndarray) -> float:
    """
    Try 0/90/180/270 to handle phone orientation differences.
    Return the minimum score.
    """
    scores = []
    x = candidate_norm
    for _ in range(4):
        scores.append(diff_score(x, reference_norm))
        x = cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE)
    return min(scores)


def main():
    ap = argparse.ArgumentParser(description="Remove duplicate photos by normalized pixel comparison (handles rotation + resolution).")
    ap.add_argument("--dir", required=True, help="Folder containing photos (searched recursively).")
    ap.add_argument("--size", type=int, default=256, help="Normalization size (e.g., 128/256).")
    ap.add_argument("--threshold", type=float, default=2.0,
                    help="Duplicate threshold on mean abs diff (typical: 1.0-4.0). Smaller = stricter.")
    ap.add_argument("--move", action="store_true", help="Move duplicates into ./duplicates instead of deleting.")
    ap.add_argument("--delete", action="store_true", help="Actually delete duplicates (danger).")
    args = ap.parse_args()

    if args.delete and args.move:
        raise SystemExit("Use only one of --move or --delete (or neither; default is --move-like behavior).")

    root = Path(args.dir).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Folder not found: {root}")

    dup_dir = root / "duplicates"
    dup_dir.mkdir(exist_ok=True)

    images = list_images(root)
    if not images:
        raise SystemExit(f"No images found under: {root}")

    # We keep a list of "unique" normalized references to compare against.
    # Simple + works well for a few thousand photos.
    kept = []  # list of tuples: (path, norm)

    moved = 0
    deleted = 0
    kept_count = 0
    unreadable = 0

    for i, path in enumerate(images, 1):
        img = load_bgr(path)
        if img is None:
            print(f"[WARN] unreadable: {path}")
            unreadable += 1
            continue

        norm = normalize(img, args.size)

        is_dup = False
        best = 999.0
        best_ref = None

        # Compare to previously kept images
        for ref_path, ref_norm in kept:
            s = best_rotation_match_score(norm, ref_norm)
            if s < best:
                best = s
                best_ref = ref_path
            if s < args.threshold:
                is_dup = True
                break

        if is_dup:
            print(f"[DUP] {path.name}  ~  {best_ref.name if best_ref else '???'}  score={best:.3f}")

            if args.delete:
                try:
                    path.unlink()
                    deleted += 1
                except Exception as e:
                    print(f"[ERROR] delete failed: {path} -> {e}")
            else:
                # default behavior: move (safer)
                target = dup_dir / path.name
                # avoid name collisions
                if target.exists():
                    stem = target.stem
                    suf = target.suffix
                    k = 1
                    while True:
                        target2 = dup_dir / f"{stem}_{k:04d}{suf}"
                        if not target2.exists():
                            target = target2
                            break
                        k += 1
                try:
                    shutil.move(str(path), str(target))
                    moved += 1
                except Exception as e:
                    print(f"[ERROR] move failed: {path} -> {e}")
        else:
            kept.append((path, norm))
            kept_count += 1

        if i % 200 == 0:
            print(f"[INFO] processed {i}/{len(images)} | kept={kept_count} moved={moved} deleted={deleted} unreadable={unreadable}")

    print("\nDone.")
    print(f"Total scanned: {len(images)}")
    print(f"Kept:         {kept_count}")
    print(f"Moved dups:   {moved}")
    print(f"Deleted dups: {deleted}")
    print(f"Unreadable:   {unreadable}")
    if not args.delete:
        print(f"Duplicates folder: {dup_dir}")


if __name__ == "__main__":
    main()