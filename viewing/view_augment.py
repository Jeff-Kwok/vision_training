#!/usr/bin/env python3
import random
import math
from pathlib import Path
import glob

import cv2
import numpy as np

# ---------------------------
# CONFIG (edit these)
# ---------------------------
IMAGES_DIR = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/labeled/dataset_4/images"
LABELS_DIR = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/labeled/dataset_4/labels"

OUT_IMAGES_DIR = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/labeled/dataset_4_aug_preview/images"
OUT_LABELS_DIR = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/labeled/dataset_4_aug_preview/labels"

SEED = 42

# How many accepted augmented variants per source image
TARGET_ACCEPTS_PER_IMAGE = 2

# Display
DISPLAY_MAX_W = 1600
DISPLAY_MAX_H = 900
WINDOW_NAME = "Aug Preview (a=accept, r=reject, n/space/-> next, p/<- prev, o=toggle, q=quit)"

CLASS_NAMES = ["bolt", "shackle"]

# geometric aug probabilities (update boxes)
P_HFLIP = 0.50
P_AFFINE = 0.60

# bbox-referenced augs
P_CROP_AROUND_OBJECT = 0.35   # updates boxes
P_OCCLUDE_IN_BBOX    = 0.30   # boxes unchanged

# photometric (boxes unchanged)
P_HSV   = 0.80
P_GAMMA = 0.25
P_BLUR  = 0.20
P_NOISE = 0.20

# affine ranges
ROT_DEG     = 12.0
SCALE_RANGE = (0.90, 1.10)
TRANS_FRAC  = 0.06

# crop around object (keep full size by resizing back)
CROP_SCALE_RANGE = (0.70, 1.00)

# occlude (cutout) inside bbox
OCCLUDE_AREA_FRAC = (0.08, 0.30)

PAD_COLOR = (114, 114, 114)
# ---------------------------


def list_images(folder: str):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(str(Path(folder) / e)))
    return sorted(files)


def read_yolo_txt(txt_path: Path):
    """Return list of (cid, cx, cy, w, h) normalized."""
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


def write_yolo_txt(txt_path: Path, boxes):
    lines = []
    for cid, cx, cy, w, h in boxes:
        cx = float(np.clip(cx, 0.0, 1.0))
        cy = float(np.clip(cy, 0.0, 1.0))
        w  = float(np.clip(w,  0.0, 1.0))
        h  = float(np.clip(h,  0.0, 1.0))
        if w <= 1e-6 or h <= 1e-6:
            continue
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def fit_to_screen(img_bgr, max_w, max_h):
    h, w = img_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return img_bgr
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)


# ---------- box math ----------
def yolo_to_xyxy(cx, cy, w, h, W, H):
    cx *= W; cy *= H; w *= W; h *= H
    x1 = cx - w/2.0
    y1 = cy - h/2.0
    x2 = cx + w/2.0
    y2 = cy + h/2.0
    return x1, y1, x2, y2


def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    x1 = np.clip(x1, 0, W-1); x2 = np.clip(x2, 0, W-1)
    y1 = np.clip(y1, 0, H-1); y2 = np.clip(y2, 0, H-1)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w/2.0
    cy = y1 + h/2.0
    return cx / W, cy / H, w / W, h / H


def clamp_boxes_yolo(boxes, eps=1e-6):
    out = []
    for cid, cx, cy, w, h in boxes:
        cx = float(np.clip(cx, 0.0, 1.0))
        cy = float(np.clip(cy, 0.0, 1.0))
        w  = float(np.clip(w,  0.0, 1.0))
        h  = float(np.clip(h,  0.0, 1.0))
        if w <= eps or h <= eps:
            continue
        out.append((cid, cx, cy, w, h))
    return out


# ---------- drawing ----------
def draw_boxes(img_bgr, boxes):
    H, W = img_bgr.shape[:2]
    out = img_bgr.copy()
    for cid, cx, cy, w, h in boxes:
        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, W, H)
        x1 = int(round(x1)); y1 = int(round(y1))
        x2 = int(round(x2)); y2 = int(round(y2))
        name = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else str(cid)

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        y_top = max(0, y1 - th - 8)
        cv2.rectangle(out, (x1, y_top), (x1 + tw + 8, y1), (0, 255, 0), -1)
        cv2.putText(out, name, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def side_by_side(left_bgr, right_bgr):
    # match heights by padding
    h1, w1 = left_bgr.shape[:2]
    h2, w2 = right_bgr.shape[:2]
    h = max(h1, h2)

    def pad_to_h(img, h_target):
        h0, w0 = img.shape[:2]
        if h0 == h_target:
            return img
        pad = h_target - h0
        top = pad // 2
        bot = pad - top
        return cv2.copyMakeBorder(img, top, bot, 0, 0, cv2.BORDER_CONSTANT, value=(30,30,30))

    L = pad_to_h(left_bgr, h)
    R = pad_to_h(right_bgr, h)
    return np.concatenate([L, R], axis=1)


# ---------- geometry transforms ----------
def apply_hflip(img, boxes):
    img2 = cv2.flip(img, 1)
    out_boxes = [(cid, 1.0 - cx, cy, w, h) for (cid, cx, cy, w, h) in boxes]
    return img2, out_boxes


def random_affine_matrix(W, H, rng):
    angle = rng.uniform(-ROT_DEG, ROT_DEG)
    scale = rng.uniform(SCALE_RANGE[0], SCALE_RANGE[1])
    tx = rng.uniform(-TRANS_FRAC, TRANS_FRAC) * W
    ty = rng.uniform(-TRANS_FRAC, TRANS_FRAC) * H
    cx, cy = W / 2.0, H / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return M


def apply_affine(img, boxes, M):
    H, W = img.shape[:2]
    img2 = cv2.warpAffine(
        img, M, (W, H), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=PAD_COLOR
    )

    out_boxes = []
    for cid, cx, cy, w, h in boxes:
        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, W, H)
        pts = np.array([[x1, y1, 1.0],
                        [x2, y1, 1.0],
                        [x2, y2, 1.0],
                        [x1, y2, 1.0]], dtype=np.float32)
        warped = (M @ pts.T).T  # 4x2
        xs = warped[:, 0]
        ys = warped[:, 1]
        nx1, ny1, nx2, ny2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

        # discard if fully outside
        if nx2 < 0 or ny2 < 0 or nx1 >= W or ny1 >= H:
            continue

        ncx, ncy, nw, nh = xyxy_to_yolo(nx1, ny1, nx2, ny2, W, H)
        if nw <= 1e-6 or nh <= 1e-6:
            continue
        out_boxes.append((cid, ncx, ncy, nw, nh))

    return img2, clamp_boxes_yolo(out_boxes)


# ---------- bbox-referenced transforms ----------
def crop_around_object_keep_size(img, boxes, rng):
    """Crop around a random object, then resize back to original size; updates boxes."""
    H, W = img.shape[:2]
    if not boxes:
        return img, boxes

    cid0, cx0, cy0, bw0, bh0 = rng.choice(boxes)
    obj_x = cx0 * W
    obj_y = cy0 * H

    scale = rng.uniform(CROP_SCALE_RANGE[0], CROP_SCALE_RANGE[1])
    crop_w = int(round(W * scale))
    crop_h = int(round(H * scale))
    crop_w = max(64, min(W, crop_w))
    crop_h = max(64, min(H, crop_h))

    x1 = int(round(obj_x - crop_w / 2))
    y1 = int(round(obj_y - crop_h / 2))

    # small jitter
    x1 += rng.randint(-int(0.10 * crop_w), int(0.10 * crop_w))
    y1 += rng.randint(-int(0.10 * crop_h), int(0.10 * crop_h))

    x1 = max(0, min(W - crop_w, x1))
    y1 = max(0, min(H - crop_h, y1))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    cropped = img[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_AREA)

    out_boxes = []
    for cid, cx, cy, bw, bh in boxes:
        bx1, by1, bx2, by2 = yolo_to_xyxy(cx, cy, bw, bh, W, H)
        # shift into crop coords
        bx1 -= x1; bx2 -= x1
        by1 -= y1; by2 -= y1

        # clip to crop
        bx1c = float(np.clip(bx1, 0, crop_w - 1))
        bx2c = float(np.clip(bx2, 0, crop_w - 1))
        by1c = float(np.clip(by1, 0, crop_h - 1))
        by2c = float(np.clip(by2, 0, crop_h - 1))

        if bx2c <= bx1c or by2c <= by1c:
            continue

        sx = W / float(crop_w)
        sy = H / float(crop_h)
        rx1 = bx1c * sx
        rx2 = bx2c * sx
        ry1 = by1c * sy
        ry2 = by2c * sy

        ncx, ncy, nw, nh = xyxy_to_yolo(rx1, ry1, rx2, ry2, W, H)
        out_boxes.append((cid, ncx, ncy, nw, nh))

    return resized, clamp_boxes_yolo(out_boxes)


def occlude_inside_bbox(img, boxes, rng):
    """Gray cutout inside a random bbox; boxes unchanged."""
    if not boxes:
        return img, boxes

    H, W = img.shape[:2]
    out = img.copy()

    cid, cx, cy, bw, bh = rng.choice(boxes)
    x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, W, H)
    x1 = int(round(x1)); y1 = int(round(y1)); x2 = int(round(x2)); y2 = int(round(y2))

    bw_px = max(1, x2 - x1)
    bh_px = max(1, y2 - y1)
    bbox_area = bw_px * bh_px

    frac = rng.uniform(OCCLUDE_AREA_FRAC[0], OCCLUDE_AREA_FRAC[1])
    occ_area = max(64, int(round(bbox_area * frac)))

    occ_w = int(round(math.sqrt(occ_area) * rng.uniform(0.7, 1.4)))
    occ_h = max(1, int(round(occ_area / max(1, occ_w))))
    occ_w = max(8, min(bw_px, occ_w))
    occ_h = max(8, min(bh_px, occ_h))

    ox1 = rng.randint(x1, max(x1, x2 - occ_w))
    oy1 = rng.randint(y1, max(y1, y2 - occ_h))
    ox2 = ox1 + occ_w
    oy2 = oy1 + occ_h

    cv2.rectangle(out, (ox1, oy1), (ox2, oy2), PAD_COLOR, -1)
    return out, boxes


# ---------- photometric transforms ----------
def hsv_jitter(img, rng):
    out = img.copy()
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_shift = rng.uniform(-8, 8)
    s_mult  = rng.uniform(0.85, 1.15)
    v_mult  = rng.uniform(0.85, 1.15)
    hsv[..., 0] = (hsv[..., 0] + h_shift) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * s_mult, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * v_mult, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def gamma_adjust(img, rng):
    gamma = rng.uniform(0.85, 1.25)
    inv = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv * 255.0
    table = table.astype(np.uint8)
    return cv2.LUT(img, table)

def maybe_blur(img, rng):
    k = rng.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)

def add_noise(img, rng):
    sigma = rng.uniform(3.0, 10.0)
    n = np.random.randn(*img.shape).astype(np.float32) * sigma
    out = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    return out


# ---------- augmentation pipeline ----------
def augment_once(img, boxes, rng):
    H, W = img.shape[:2]
    out_img = img
    out_boxes = list(boxes)

    # bbox-referenced crop (updates boxes)
    if rng.random() < P_CROP_AROUND_OBJECT:
        out_img, out_boxes = crop_around_object_keep_size(out_img, out_boxes, rng)

    # geometric: flip
    if rng.random() < P_HFLIP:
        out_img, out_boxes = apply_hflip(out_img, out_boxes)

    # geometric: affine
    if rng.random() < P_AFFINE:
        M = random_affine_matrix(W, H, rng)
        out_img, out_boxes = apply_affine(out_img, out_boxes, M)

    # occlude inside bbox (boxes unchanged)
    if rng.random() < P_OCCLUDE_IN_BBOX:
        out_img, out_boxes = occlude_inside_bbox(out_img, out_boxes, rng)

    # photometric
    if rng.random() < P_HSV:
        out_img = hsv_jitter(out_img, rng)
    if rng.random() < P_GAMMA:
        out_img = gamma_adjust(out_img, rng)
    if rng.random() < P_BLUR:
        out_img = maybe_blur(out_img, rng)
    if rng.random() < P_NOISE:
        out_img = add_noise(out_img, rng)

    out_boxes = clamp_boxes_yolo(out_boxes)
    return out_img, out_boxes


# ---------- save helpers ----------
def next_available_stem(out_img_dir: Path, base_stem: str):
    """Find a unique stem to avoid overwriting."""
    k = 0
    while True:
        stem = f"{base_stem}__aug{k:03d}"
        if not (out_img_dir / f"{stem}.jpg").exists():
            return stem
        k += 1


def main():
    rng = random.Random(SEED)

    images = list_images(IMAGES_DIR)
    if not images:
        raise SystemExit(f"No images found in {IMAGES_DIR}")

    images = [Path(p) for p in images]
    labels_dir = Path(LABELS_DIR)

    out_img_dir = Path(OUT_IMAGES_DIR)
    out_lbl_dir = Path(OUT_LABELS_DIR)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Track how many accepted per source
    accepted_counts = {p.stem: 0 for p in images}

    idx = 0
    show_side_by_side = True

    while 0 <= idx < len(images):
        img_path = images[idx]
        txt_path = labels_dir / f"{img_path.stem}.txt"

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] cannot read {img_path}")
            idx += 1
            continue

        boxes = read_yolo_txt(txt_path)

        # If already reached target accepts, auto-advance
        if accepted_counts[img_path.stem] >= TARGET_ACCEPTS_PER_IMAGE:
            idx += 1
            continue

        # Generate a candidate augmentation
        aug_img, aug_boxes = augment_once(img, boxes, rng)

        # Draw overlays
        orig_vis = draw_boxes(img, boxes)
        aug_vis  = draw_boxes(aug_img, aug_boxes)

        # Compose preview
        if show_side_by_side:
            preview = side_by_side(orig_vis, aug_vis)
        else:
            preview = aug_vis

        # Text overlay
        info1 = f"{idx+1}/{len(images)}  {img_path.name}"
        info2 = f"accepted {accepted_counts[img_path.stem]}/{TARGET_ACCEPTS_PER_IMAGE}  boxes={len(boxes)}"
        info3 = "a=accept  r=reject  n/space/-> next  p/<- prev  o=toggle  q=quit"
        cv2.putText(preview, info1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(preview, info2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(preview, info3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        disp = fit_to_screen(preview, DISPLAY_MAX_W, DISPLAY_MAX_H)
        cv2.imshow(WINDOW_NAME, disp)

        key = cv2.waitKey(0) & 0xFFFFFFFF
        LEFT_ARROW  = 81
        RIGHT_ARROW = 83

        if key in (ord('q'), 27):
            break

        elif key in (ord('o'),):
            show_side_by_side = not show_side_by_side

        elif key in (ord('r'),):
            # reject: do nothing; regenerate a new candidate for same image
            continue

        elif key in (ord('n'), ord(' '), RIGHT_ARROW, 2555904):
            idx += 1

        elif key in (ord('p'), LEFT_ARROW, 2424832):
            idx = max(0, idx - 1)

        elif key == ord('a'):
            # accept: save augmented image + yolo txt
            stem = next_available_stem(out_img_dir, img_path.stem)
            out_img_path = out_img_dir / f"{stem}.jpg"
            out_lbl_path = out_lbl_dir / f"{stem}.txt"

            ok = cv2.imwrite(str(out_img_path), aug_img)
            if not ok:
                print(f"[WARN] failed to write {out_img_path}")
                continue

            write_yolo_txt(out_lbl_path, aug_boxes)

            accepted_counts[img_path.stem] += 1
            print(f"[SAVED] {out_img_path.name}  (from {img_path.name})  count={accepted_counts[img_path.stem]}")

            # stay on same source until it reaches TARGET_ACCEPTS_PER_IMAGE
            continue

        else:
            # default: advance
            idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()