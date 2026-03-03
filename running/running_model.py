#!/usr/bin/env python3
# This runs the models and lets you confirm the whether the annotated image is correct or not. Best to feed new data.
import glob
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import supervision as sv

from rfdetr import RFDETRMedium

# -------------------
# Config
# -------------------
IMAGES_DIR = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/scripts/stable/running/testing"
OUT_DIR    = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/raw/photos/raw_consolidated/standardized_1120_labeled"

WEIGHTS    = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/results/model_7_0/checkpoint_best_total.pth"
THRESHOLD  = 0.80

CLASS_NAMES = ["bolt", "shackle"]  # 0 bolt, 1 shackle

WINDOW_NAME = "RF-DETR review (a=accept, r=refuse, e=edit, s=skip, q=quit)"
DISPLAY_MAX_W = 1200
DISPLAY_MAX_H = 800
# -------------------


def list_images(folder: str):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(str(Path(folder) / e)))
    return sorted(files)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    x1 = clamp(x1, 0, W - 1)
    y1 = clamp(y1, 0, H - 1)
    x2 = clamp(x2, 0, W - 1)
    y2 = clamp(y2, 0, H - 1)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return (cx / W, cy / H, bw / W, bh / H)


def write_yolo_labels(txt_path: Path, det: sv.Detections, W: int, H: int):
    lines = []
    if det is None or len(det) == 0:
        txt_path.write_text("")  # empty file is valid
        return
    for (x1, y1, x2, y2), cid in zip(det.xyxy, det.class_id):
        cid = int(cid)
        cx, cy, w, h = xyxy_to_yolo(float(x1), float(y1), float(x2), float(y2), W, H)
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def fit_for_display(img_bgr, max_w, max_h):
    h, w = img_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return img_bgr
    return cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def ensure_out_dirs(base: Path):
    (base / "images").mkdir(parents=True, exist_ok=True)
    (base / "labels").mkdir(parents=True, exist_ok=True)
    (base / "refuse" / "images").mkdir(parents=True, exist_ok=True)
    (base / "refuse" / "labels").mkdir(parents=True, exist_ok=True)


def save_pair(dest_root: Path, img_path: Path, frame_rgb: np.ndarray, det: sv.Detections, W: int, H: int):
    """
    dest_root: OUT_DIR or OUT_DIR/refuse
      writes:
        dest_root/images/<original filename>
        dest_root/labels/<stem>.txt
    """
    out_img = dest_root / "images" / img_path.name
    out_txt = dest_root / "labels" / f"{img_path.stem}.txt"
 
    cv2.imwrite(str(out_img), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    write_yolo_labels(out_txt, det, W, H)
    return out_img, out_txt


def edit_boxes_with_opencv(frame_rgb):
    """
    Draw fresh boxes using OpenCV selectROIs and choose class id per ROI via terminal input.
    Returns sv.Detections.
    """
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    rois = cv2.selectROIs(
        "Edit: draw boxes, ENTER=finish, ESC=cancel",
        frame_bgr,
        fromCenter=False,
        showCrosshair=True
    )
    cv2.destroyWindow("Edit: draw boxes, ENTER=finish, ESC=cancel")

    if rois is None or len(rois) == 0:
        return sv.Detections.empty()

    new_xyxy, new_cids, new_confs = [], [], []

    print("\n[EDIT] For each ROI, enter class id (0=bolt, 1=shackle).")
    for i, (x, y, w, h) in enumerate(rois):
        x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
        while True:
            k = input(f"ROI {i+1}/{len(rois)} class (0/1): ").strip()
            if k in ("0", "1"):
                cid = int(k)
                break
            print("  Please enter 0 or 1.")
        new_xyxy.append([x1, y1, x2, y2])
        new_cids.append(cid)
        new_confs.append(1.0)

    return sv.Detections(
        xyxy=np.array(new_xyxy, dtype=np.float32),
        confidence=np.array(new_confs, dtype=np.float32),
        class_id=np.array(new_cids, dtype=np.int32),
    )


def main():
    img_files = list_images(IMAGES_DIR)
    if not img_files:
        raise SystemExit(f"No images found in {IMAGES_DIR}")

    out_dir = Path(OUT_DIR)
    ensure_out_dirs(out_dir)

    print(f"[INFO] Loading RF-DETR weights: {WEIGHTS}")
    model = RFDETRMedium(pretrain_weights=WEIGHTS)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    idx = 0
    while idx < len(img_files):
        path = Path(img_files[idx])
        img_pil = Image.open(path).convert("RGB")
        W, H = img_pil.size
        frame_rgb = np.array(img_pil)

        det = model.predict(img_pil, threshold=THRESHOLD)

        # labels for display
        labels = [CLASS_NAMES[int(cid)] for cid in det.class_id] if det is not None else []

        annotated_rgb = box_annotator.annotate(frame_rgb.copy(), det)
        annotated_rgb = label_annotator.annotate(annotated_rgb, det, labels)

        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(
            annotated_bgr,
            f"{idx+1}/{len(img_files)}  a=accept  r=refuse  e=edit  s=skip  q=quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        display_img = fit_for_display(annotated_bgr, DISPLAY_MAX_W, DISPLAY_MAX_H)
        cv2.imshow(WINDOW_NAME, display_img)

        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):
            print(f"[SKIP] {path.name}")
            idx += 1
            continue

        elif key == ord("a"):
            out_img, out_txt = save_pair(out_dir, path, frame_rgb, det, W, H)
            print(f"[ACCEPT] {path.name} -> {out_img} + {out_txt}")
            idx += 1
            continue

        elif key == ord("r"):
            refuse_root = out_dir / "refuse"
            out_img, out_txt = save_pair(refuse_root, path, frame_rgb, det, W, H)
            print(f"[REFUSE] {path.name} -> {out_img} + {out_txt}")
            idx += 1
            continue

        elif key == ord("e"):
            edited_det = edit_boxes_with_opencv(frame_rgb)

            # show edited result quickly
            edited_labels = [CLASS_NAMES[int(cid)] for cid in edited_det.class_id] if len(edited_det) else []
            tmp = box_annotator.annotate(frame_rgb.copy(), edited_det)
            tmp = label_annotator.annotate(tmp, edited_det, edited_labels)
            cv2.imshow(WINDOW_NAME, fit_for_display(cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR), DISPLAY_MAX_W, DISPLAY_MAX_H))

            print("[EDIT] Press 'a' to save edited as ACCEPT, 'r' to save edited as REFUSE, any other key to cancel.")
            k2 = cv2.waitKey(0) & 0xFF

            if k2 == ord("a"):
                out_img, out_txt = save_pair(out_dir, path, frame_rgb, edited_det, W, H)
                print(f"[ACCEPT-EDITED] {path.name} -> {out_img} + {out_txt}")
                idx += 1
            elif k2 == ord("r"):
                refuse_root = out_dir / "refuse"
                out_img, out_txt = save_pair(refuse_root, path, frame_rgb, edited_det, W, H)
                print(f"[REFUSE-EDITED] {path.name} -> {out_img} + {out_txt}")
                idx += 1
            else:
                print("[EDIT] Canceled (nothing saved).")

            continue

        else:
            print("[INFO] Unknown key. Use a/r/e/s/q.")
            # stay on same image

    cv2.destroyAllWindows()
    print(f"[DONE] Saved dataset to: {OUT_DIR}")
    print(f"       accept: {OUT_DIR}/images + {OUT_DIR}/labels")
    print(f"       refuse: {OUT_DIR}/refuse/images + {OUT_DIR}/refuse/labels")
    print("       Class mapping: 0 bolt, 1 shackle")


if __name__ == "__main__":
    main()
