#!/usr/bin/env python3
import glob
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageOps

# -------------------
# Config (EDIT THESE)
# -------------------
IMAGES_DIR = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/raw/photos/raw_consolidated/batch4_feb25"

# Refused originals get moved here
REMOVED_DIR = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/raw/photos/raw_consolidated/removed"

# Accepted standardized outputs written here (does NOT move original)
ACCEPTED_DIR = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/raw/photos/raw_consolidated/standardized_1120"

DISPLAY_MAX_W = 1080
DISPLAY_MAX_H = 720

# Training input size (requested)
TARGET_W = 1120
TARGET_H = 1120

# Letterbox padding color (BGR). Use 114-ish for a neutral gray.
PAD_COLOR = (114, 114, 114)

JPEG_QUALITY = 92

WINDOW_NAME = "QC Viewer (->/n/space next, <-/p prev, a accept, d refuse, q/esc quit)"
# -------------------


def list_images(folder: str):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(str(Path(folder) / e)))
    return sorted(files)


def fit_to_screen(img_bgr, max_w, max_h):
    """Resize only for display."""
    h, w = img_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return img_bgr
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def load_bgr_with_exif_fix(path: Path) -> np.ndarray:
    """
    Load via PIL to apply EXIF orientation correctly, then convert to OpenCV BGR.
    """
    pil = Image.open(path)
    pil = ImageOps.exif_transpose(pil)  # rotate/flip according to EXIF
    pil = pil.convert("RGB")
    rgb = np.array(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def letterbox_to(img_bgr: np.ndarray, target_w: int, target_h: int, pad_color=(114, 114, 114)):
    """
    Resize (keeping aspect) then pad to exact target size.
    Returns: (out_img, scale, pad_left, pad_top)
    """
    h, w = img_bgr.shape[:2]

    # scale so it fits inside target
    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w = target_w - new_w
    pad_h = target_h - new_h

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color
    )
    return out, scale, left, top


def safe_move(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dest = dst_dir / src.name

    if dest.exists():
        stem = dest.stem
        suf = dest.suffix
        k = 1
        while True:
            new_dest = dst_dir / f"{stem}_{k:04d}{suf}"
            if not new_dest.exists():
                dest = new_dest
                break
            k += 1

    shutil.move(str(src), str(dest))
    return dest


def safe_write_jpg(img_bgr: np.ndarray, out_dir: Path, stem: str, quality: int = 92) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.jpg"

    if out_path.exists():
        k = 1
        while True:
            candidate = out_dir / f"{stem}_{k:04d}.jpg"
            if not candidate.exists():
                out_path = candidate
                break
            k += 1

    ok = cv2.imwrite(str(out_path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imwrite failed")
    return out_path


def main():
    images = list_images(IMAGES_DIR)
    if not images:
        raise SystemExit(f"No images found in {IMAGES_DIR}")

    Path(REMOVED_DIR).mkdir(parents=True, exist_ok=True)
    Path(ACCEPTED_DIR).mkdir(parents=True, exist_ok=True)

    idx = 0
    while 0 <= idx < len(images):
        img_path = Path(images[idx])

        try:
            # Load with EXIF orientation fix
            img_bgr = load_bgr_with_exif_fix(img_path)
        except Exception as e:
            print(f"[WARN] Could not load {img_path}: {e}")
            idx += 1
            continue

        H, W = img_bgr.shape[:2]

        # Build standardized preview (1120x1120) for reference
        std_bgr, scale, pad_l, pad_t = letterbox_to(img_bgr, TARGET_W, TARGET_H, PAD_COLOR)

        # Display original (not standardized) but show info
        disp = fit_to_screen(std_bgr, DISPLAY_MAX_W, DISPLAY_MAX_H)
        count = len(images)

        # Overlay info
        cv2.putText(disp, f"{idx+1}/{count}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(disp, f"{idx+1}/{count}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(disp, f"orig: {W}x{H}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(disp, f"orig: {W}x{H}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(disp, f"train: {TARGET_W}x{TARGET_H} (letterbox)", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(disp, f"train: {TARGET_W}x{TARGET_H} (letterbox)", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, disp)
        key = cv2.waitKey(0) & 0xFFFFFFFF

        LEFT_ARROW = 81
        RIGHT_ARROW = 83

        if key in (ord('q'), 27):
            break

        elif key in (RIGHT_ARROW, 2555904, ord('n'), ord(' ')):  # next
            idx += 1

        elif key in (LEFT_ARROW, 2424832, ord('p')):  # previous
            idx = max(0, idx - 1)

        elif key == ord('a'):  # ACCEPT -> write standardized 1120x1120
            out_path = safe_write_jpg(std_bgr, Path(ACCEPTED_DIR), img_path.stem, JPEG_QUALITY)
            print(f"[ACCEPTED] {img_path.name} -> {out_path.name}")
            idx += 1

        elif key == ord('d'):  # REFUSE -> move original
            moved_to = safe_move(img_path, Path(REMOVED_DIR))
            print(f"[REMOVED] {img_path.name} -> {moved_to}")

            images.pop(idx)
            if idx >= len(images):
                idx = len(images) - 1

        else:
            idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()