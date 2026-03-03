#!/usr/bin/env python3
import cv2 as cv
import numpy as np
from rfdetr import RFDETRMedium
import supervision as sv

WEIGHTS = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/results/model_7_0/checkpoint_best_total.pth"
GROUND_TRUTH_OPEN = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/scripts/stable/running/testing/ground_truth_open.png"

# RF-DETR
THRESHOLD   = 0.90
NUM_CLASSES = 2
CLASS_NAMES = ["bolt", "shackle"]

# Your current (fragile) mask generator; keep it for now
dark_gray  = (0, 0, 125)
light_gray = (179, 130, 255)

# -------------------------
# Mask utilities
# -------------------------
def binarize(mask):
    """Return 0/255 uint8 binary mask."""
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    _, m = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    return m

def pad_to_square(mask, pad_value=0):
    h, w = mask.shape[:2]
    s = max(h, w)
    out = np.full((s, s), pad_value, dtype=mask.dtype)
    y0 = (s - h) // 2
    x0 = (s - w) // 2
    out[y0:y0+h, x0:x0+w] = mask
    return out

def resize_mask(mask, size=(256, 256)):
    return cv.resize(mask, size, interpolation=cv.INTER_NEAREST)

def centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return None
    return (xs.mean(), ys.mean())

def shift_mask(mask, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv.warpAffine(mask, M, (mask.shape[1], mask.shape[0]),
                         flags=cv.INTER_NEAREST, borderValue=0)

def rotate_mask(mask, angle_deg):
    """Rotate binary mask around center."""
    h, w = mask.shape[:2]
    M = cv.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv.warpAffine(mask, M, (w, h), flags=cv.INTER_NEAREST, borderValue=0)

def dice(mask_a, mask_b):
    a = (mask_a > 0)
    b = (mask_b > 0)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float((2 * inter) / denom) if denom > 0 else 0.0

def iou(mask_a, mask_b):
    a = (mask_a > 0)
    b = (mask_b > 0)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0

def normalize_for_match(mask, size=(256, 256)):
    """Binarize -> pad to square -> resize to fixed size."""
    m = binarize(mask)
    if m is None:
        return None
    m = pad_to_square(m)
    m = resize_mask(m, size)
    return m

def overlay_match(live_aligned, gt_norm):
    """Green overlap, red live-only, blue gt-only."""
    overlay = np.zeros((gt_norm.shape[0], gt_norm.shape[1], 3), dtype=np.uint8)
    L = live_aligned > 0
    G = gt_norm > 0
    overlay[L & ~G] = (0, 0, 255)
    overlay[~L & G] = (255, 0, 0)
    overlay[L & G]  = (0, 255, 0)
    return overlay

# -------------------------
# Rotation-invariant matcher
# -------------------------
def match_masks_rot_invariant(mask_live, gt_norm, step_deg=4, search_deg=360):
    """
    Find rotation (relative to gt_norm) that maximizes Dice overlap.
    Returns: (bestDice, bestIoU, bestAngleDeg, bestLiveAligned, overlay)
    bestAngleDeg is the rotation applied to the LIVE mask to match the GT reference.
    """
    live = normalize_for_match(mask_live, size=(gt_norm.shape[1], gt_norm.shape[0]))
    if live is None:
        return 0.0, 0.0, 0.0, None, None

    bestD, bestI, bestA, bestM = -1.0, 0.0, 0.0, None

    for a in range(0, search_deg, step_deg):
        m = rotate_mask(live, a)

        # centroid align after rotation
        c1 = centroid(m)
        c2 = centroid(gt_norm)
        if c1 is not None and c2 is not None:
            dx = int(round(c2[0] - c1[0]))
            dy = int(round(c2[1] - c1[1]))
            m = shift_mask(m, dx, dy)

        D = dice(m, gt_norm)
        if D > bestD:
            bestD = D
            bestI = iou(m, gt_norm)
            bestA = float(a)
            bestM = m

    ov = overlay_match(bestM, gt_norm) if bestM is not None else None
    return bestD, bestI, bestA, bestM, ov

# -------------------------
# Your current mask extractor (still HSV)
# -------------------------
def extract_pose_mask(crop_bgr):
    """
    Returns (mask, contours) where mask is 0/255 uint8.
    """
    crop_blur = cv.GaussianBlur(crop_bgr, (3, 3), 0)
    hsv = cv.cvtColor(crop_blur, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsv, dark_gray, light_gray)
    mask = cv.bitwise_not(mask)

    # cleanup helps stability
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return mask, contours

# -------------------------
# Per-frame processing
# -------------------------
def process_frame(frame_bgr, det, gt_norm):
    out = frame_bgr.copy()

    best = None
    # best = (dice, iou, rel_rot_deg, x1,y1,x2,y2, class_id, conf, rect_pts_full, live_mask, live_aligned, overlay)

    for box, class_id, conf in zip(det.xyxy, det.class_id, det.confidence):
        x1, y1, x2, y2 = np.round(box).astype(int)

        H, W = out.shape[:2]
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame_bgr[y1:y2, x1:x2].copy()
        live_mask, contours = extract_pose_mask(crop)

        # draw bbox lightly for context
        cv.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)

        if contours is None or len(contours) == 0:
            continue

        c = max(contours, key=cv.contourArea)
        if cv.contourArea(c) < 200:
            continue

        # rotated rect (for visualization only)
        rect = cv.minAreaRect(c)
        rect_pts = cv.boxPoints(rect)
        rect_pts = np.round(rect_pts).astype(int)
        rect_pts_full = rect_pts + np.array([x1, y1])

        # rotation-invariant match against GT open-shackle reference
        D, I, rel_rot_deg, live_aligned, ov = match_masks_rot_invariant(
            live_mask, gt_norm, step_deg=4, search_deg=360
        )

        if best is None or D > best[0]:
            best = (D, I, rel_rot_deg, x1, y1, x2, y2, int(class_id), float(conf),
                    rect_pts_full, live_mask, live_aligned, ov)

    # highlight best match
    if best is not None:
        D, I, rel_rot_deg, x1, y1, x2, y2, class_id, conf, rect_pts_full, live_mask, live_aligned, ov = best

        cv.drawContours(out, [rect_pts_full], 0, (0, 0, 255), 2)

        # rel_rot_deg = rotation applied to LIVE to match GT.
        # If you want "live relative to reference" in the opposite direction, use (-rel_rot_deg) % 360.
        label = f"{CLASS_NAMES[class_id]} {conf:.2f} | Dice {D:.3f} IoU {I:.3f} | relRot {rel_rot_deg:.0f}deg"
        cv.putText(out, label, (x1, max(0, y1 - 8)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1, cv.LINE_AA)

        return out, live_mask, live_aligned, ov

    return out, None, None, None

# -------------------------
# Main
# -------------------------
def main():
    # Load ground truth thresholded mask (open shackle reference)
    gt = cv.imread(GROUND_TRUTH_OPEN, cv.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(f"Could not read ground truth mask: {GROUND_TRUTH_OPEN}")

    gt_norm = normalize_for_match(gt, size=(256, 256))

    cap = cv.VideoCapture(4)
    model = RFDETRMedium(pretrain_weights=WEIGHTS, num_classes=NUM_CLASSES)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        det = model.predict(frame, threshold=THRESHOLD)

        annotated, live_mask, live_aligned, overlay = process_frame(frame, det, gt_norm)

        cv.imshow("annotated", annotated)
        cv.imshow("gt_norm_256", gt_norm)

        if live_mask is not None:
            cv.imshow("live_mask_crop", live_mask)
        if live_aligned is not None:
            cv.imshow("live_aligned_256", live_aligned)
        if overlay is not None:
            cv.imshow("match_overlay", overlay)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()