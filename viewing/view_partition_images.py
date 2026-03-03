#!/usr/bin/env python3
import glob
from pathlib import Path

import cv2

# -------------------
# EDIT THESE
# -------------------
DATASET_ROOT = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/training_sets/dataset_8"
CLASS_NAMES = ["bolt", "shackle"]
DISPLAY_MAX_W = 1400
DISPLAY_MAX_H = 900
WINDOW_NAME = "Split Viewer (1=train 2=val 3=test | n/space next | p prev | q quit)"
# -------------------

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def list_images(folder: Path):
    files = []
    for e in IMAGE_EXTS:
        files.extend(glob.glob(str(folder / e)))
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


def load_split(root: Path, split: str):
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    imgs = list_images(img_dir)
    return img_dir, lbl_dir, imgs


def main():
    root = Path(DATASET_ROOT)

    split = "train"
    img_dir, lbl_dir, imgs = load_split(root, split)
    if not imgs:
        print(f"[WARN] No images found in {img_dir}")

    idx = 0
    while True:
        if not imgs:
            # show empty screen message
            blank = 255 * (cv2.UMat(200, 900, cv2.CV_8UC3).get())
            cv2.putText(blank, f"No images in split: {split}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(blank, "Press 1(train) 2(val) 3(test) or q to quit", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, blank)
            key = cv2.waitKey(0) & 0xFF
        else:
            idx = max(0, min(idx, len(imgs) - 1))
            img_path = Path(imgs[idx])
            lbl_path = lbl_dir / f"{img_path.stem}.txt"

            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] Could not read {img_path}")
                idx += 1
                continue

            boxes = read_yolo_txt(lbl_path)
            drawn = draw_boxes(img, boxes)

            info1 = f"{split.upper()}  {idx+1}/{len(imgs)}  {img_path.name}  boxes={len(boxes)}"
            info2 = "1=train 2=val 3=test | n/space next | p prev | q/esc quit"
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
            idx -= 1
        elif key == ord('1'):
            split = "train"
            img_dir, lbl_dir, imgs = load_split(root, split)
            idx = 0
        elif key == ord('2'):
            split = "valid"
            img_dir, lbl_dir, imgs = load_split(root, split)
            idx = 0
        elif key == ord('3'):
            split = "test"
            img_dir, lbl_dir, imgs = load_split(root, split)
            idx = 0
        else:
            idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
