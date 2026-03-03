#!/usr/bin/env python3
import glob
from pathlib import Path

import cv2

# -------------------
# Config (EDIT THESE)
# -------------------
IMAGES_DIR = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/labeled/dataset_consolidated/images"
LABELS_DIR = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/labeled/dataset_consolidated/labels"

CLASS_NAMES = ["bolt", "shackle"]  # 0 bolt, 1 shackle

DISPLAY_MAX_W = 1400
DISPLAY_MAX_H = 900

WINDOW_NAME = "YOLO Viewer (n/space=next, p=prev, q/esc=quit)"
# -------------------


def list_images(folder: str):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(str(Path(folder) / e)))
    return sorted(files)


def read_yolo_txt(txt_path: Path):
    """YOLO txt lines: class cx cy w h (normalized)."""
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
    """Convert normalized YOLO (cx,cy,w,h) -> pixel xyxy."""
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
    """Draw rectangles + class label text."""
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
    """Resize only for display."""
    H, W = img_bgr.shape[:2]
    scale = min(max_w / W, max_h / H, 1.0)
    if scale >= 1.0:
        return img_bgr
    new_w = int(W * scale)
    new_h = int(H * scale)
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main():
    images = list_images(IMAGES_DIR)
    if not images:
        raise SystemExit(f"No images found in {IMAGES_DIR}")

    labels_dir = Path(LABELS_DIR)

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

        info = f"{idx+1}/{len(images)}  {img_path.name}  boxes={len(boxes)}"
        cv2.putText(drawn, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        disp = fit_to_screen(drawn, DISPLAY_MAX_W, DISPLAY_MAX_H)
        cv2.imshow(WINDOW_NAME, disp)

        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key in (ord('n'), ord(' ')):
            idx += 1
        elif key == ord('p'):
            idx = max(0, idx - 1)
        else:
            idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
