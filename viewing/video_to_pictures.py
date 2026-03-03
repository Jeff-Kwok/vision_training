#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2 as cv

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

def extract_2fps(video_path: Path, out_dir: Path, fps_out: float = 2.0, jpeg_quality: int = 95):
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps_in = cap.get(cv.CAP_PROP_FPS)
    if not fps_in or fps_in <= 0:
        # fallback: assume 30fps if metadata missing
        fps_in = 30.0

    # Write one frame every N input frames
    step = max(1, int(round(fps_in / fps_out)))

    base = video_path.stem
    frame_idx = 0
    saved = 0

    print(f"[{video_path.name}] fps_in={fps_in:.3f}, target={fps_out}, step={step} (save every {step} frames)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            # timestamp (ms) from video helps keep names unique/ordered
            t_ms = cap.get(cv.CAP_PROP_POS_MSEC)
            out_path = out_dir / f"{base}_t{int(t_ms):010d}ms_f{frame_idx:09d}.jpg"
            cv.imwrite(str(out_path), frame, [int(cv.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"Saved {saved} frames -> {out_dir}")

def iter_videos(path: Path):
    if path.is_file():
        yield path
        return
    for p in sorted(path.rglob("*")):
        if p.suffix.lower() in VIDEO_EXTS:
            yield p

def main():
    ap = argparse.ArgumentParser(description="Extract frames from video(s) at 2 FPS.")
    ap.add_argument("input", type=str, help="Video file or folder containing videos")
    ap.add_argument("--out", type=str, default="frames_2fps", help="Output folder")
    ap.add_argument("--fps", type=float, default=2.0, help="Target FPS to extract (default: 2)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality 0-100 (default: 95)")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    vids = list(iter_videos(in_path))
    if not vids:
        raise SystemExit(f"No videos found at: {in_path}")

    for vp in vids:
        # Put each video's frames in its own subfolder
        out_dir = out_root / vp.stem
        extract_2fps(vp, out_dir, fps_out=args.fps, jpeg_quality=args.quality)

if __name__ == "__main__":
    main()