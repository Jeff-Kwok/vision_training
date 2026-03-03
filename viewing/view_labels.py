#!/usr/bin/env python3
import glob
import shutil
from pathlib import Path

import cv2

# -------------------
# Config (EDIT THESE)
# -------------------
IMAGES_DIR = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/raw/photos/raw_consolidated/standardized_1120_labeled/images"
LABELS_DIR = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/raw/photos/raw_consolidated/standardized_1120_labeled/labels"

APPROVED_IMAGES_DIR = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/labeled/dataset_consolidated/images"
APPROVED_LABELS_DIR = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/labeled/dataset_consolidated/labels"

BAD_OUT_DIR = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/labeled/dataset_consolidated/refuse"  # or None

CLASS_NAMES = ["bolt", "shackle"]

DISPLAY_MAX_W = 1400
DISPLAY_MAX_H = 900

WINDOW_NAME = "YOLO Label Viewer"
# -------------------


def list_images(folder: str):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(str(Path(folder) / e)))
    return sorted(files)


def read_yolo_txt(txt_path: Path):
    if not txt_path.exists():
        return []
    lines = txt_path.read_text().strip().splitlines()
    boxes = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append((cid, cx, cy, w, h))
    return boxes


def yolo_to_xyxy(cx, cy, w, h, W, H):
    cx *= W
    cy *= H
    w *= W
    h *= H
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    return x1, y1, x2, y2


def draw_boxes(img_bgr, boxes):
    H, W = img_bgr.shape[:2]
    out = img_bgr.copy()

    for cid, cx, cy, w, h in boxes:
        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, W, H)
        name = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else str(cid)

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        y_top = max(0, y1 - th - 8)
        cv2.rectangle(out, (x1, y_top), (x1 + tw + 8, y1), (0, 255, 0), -1)
        cv2.putText(out, name, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def fit_to_screen(img_bgr, max_w, max_h):
    H, W = img_bgr.shape[:2]
    scale = min(max_w / W, max_h / H, 1.0)
    if scale >= 1.0:
        return img_bgr
    new_w = int(W * scale)
    new_h = int(H * scale)
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def copy_if_missing(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return False
    shutil.copy2(src, dst)
    return True


def approve_item(img_path: Path, labels_dir: Path, out_images: Path, out_labels: Path):
    label_src = labels_dir / f"{img_path.stem}.txt"
    img_dst = out_images / img_path.name
    lbl_dst = out_labels / label_src.name

    img_copied = copy_if_missing(img_path, img_dst)

    lbl_copied = False
    lbl_missing = False
    if label_src.exists():
        lbl_copied = copy_if_missing(label_src, lbl_dst)
    else:
        lbl_missing = True

    return img_dst, lbl_dst, img_copied, lbl_copied, lbl_missing


def mark_bad(img_path: Path, bad_dir: Path):
    dst = bad_dir / img_path.name
    copied = copy_if_missing(img_path, dst)
    return dst, copied


def main():
    images = list_images(IMAGES_DIR)
    if not images:
        raise SystemExit(f"No images found in {IMAGES_DIR}")

    labels_dir = Path(LABELS_DIR)
    approved_images = Path(APPROVED_IMAGES_DIR)
    approved_labels = Path(APPROVED_LABELS_DIR)
    bad_dir = Path(BAD_OUT_DIR) if BAD_OUT_DIR else None

    idx = 0
    while 0 <= idx < len(images):
        img_path = Path(images[idx])
        txt_path = labels_dir / f"{img_path.stem}.txt"

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] Could not read {img_path}")
            idx += 1
            continue

        boxes = read_yolo_txt(txt_path)
        drawn = draw_boxes(img_bgr, boxes)

        # overlay info + controls
        info1 = f"{idx+1}/{len(images)}  {img_path.name}  boxes={len(boxes)}"
        info2 = "keys: a=approve  b=bad  n/space=next  p=prev  q/esc=quit"
        cv2.putText(drawn, info1, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(drawn, info2, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        disp = fit_to_screen(drawn, DISPLAY_MAX_W, DISPLAY_MAX_H)
        cv2.imshow(WINDOW_NAME, disp)

        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key in (ord('n'), ord(' ')):
            idx += 1
        elif key == ord('p'):
            idx = max(0, idx - 1)

        elif key == ord('a'):
            img_dst, lbl_dst, img_copied, lbl_copied, lbl_missing = approve_item(
                img_path, labels_dir, approved_images, approved_labels
            )

            print(f"[APPROVE] {img_path.name}")
            print(f"  image: {'copied' if img_copied else 'exists'} -> {img_dst}")
            if lbl_missing:
                print(f"  label: MISSING (expected {txt_path})")
            else:
                print(f"  label: {'copied' if lbl_copied else 'exists'} -> {lbl_dst}")

            idx += 1

        elif key == ord('b'):
            if bad_dir is None:
                print("[BAD] BAD_OUT_DIR is None (disabled).")
            else:
                dst, copied = mark_bad(img_path, bad_dir)
                print(f"[BAD] {img_path.name} -> {'copied' if copied else 'exists'} -> {dst}")
            idx += 1

        else:
            idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
