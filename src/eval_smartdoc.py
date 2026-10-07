"""Evaluate page detection on SmartDoc (ICDAR 2015, Challenge 1).

SmartDoc gives smartphone videos of documents plus, for every frame, the true
page corners (XML). This script samples frames from each video, runs
detect_page(), and compares the found corners with the true ones.

Layout it expects (same for the sample and the full test set):
    <root>/input_sample/<background>/<doc>.avi
    <root>/input_sample_groundtruth/<background>_gt/<doc>.gt.xml

Metrics per frame
    corner errors   pixel distance of each corner from the true corner
    IoU             overlap of the found page and the true page (1.0 = perfect)
    correct         every corner within --tol of the image diagonal (default 2%)
    clean           detector reported a clean 4-point contour (found=True)

Usage
    python -m src.eval_smartdoc
    python -m src.eval_smartdoc --step 5 --out results/smartdoc
    python -m src.eval_smartdoc --root D:\\path\\to\\testDataset
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.detect import detect_page
from src.warp import corner_errors, corners_correct, quad_iou

VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv")


def load_gt(xml_path: Path) -> dict[int, np.ndarray]:
    """Return {frame_index (1-based): corners (4,2) as TL, TR, BR, BL}.

    Frames marked rejected="true", or with missing points, are skipped.
    """
    frames = {}
    root = ET.parse(xml_path).getroot()
    for fr in root.iter("frame"):
        if str(fr.get("rejected", "false")).lower() == "true":
            continue
        pts = {p.get("name"): (float(p.get("x")), float(p.get("y"))) for p in fr.iter("point")}
        if not {"tl", "tr", "br", "bl"} <= set(pts):
            continue
        frames[int(fr.get("index"))] = np.array(
            [pts["tl"], pts["tr"], pts["br"], pts["bl"]], dtype=np.float32)
    return frames


def find_pairs(root: Path, video_dir: Path | None, gt_dir: Path | None):
    """Match every ground-truth XML with its video. Returns [(background, doc, video, xml)]."""
    video_dir = video_dir or next((d for d in root.iterdir() if d.is_dir() and d.name.startswith("input")
                                   and "groundtruth" not in d.name), None)
    gt_dir = gt_dir or next((d for d in root.iterdir() if d.is_dir() and "groundtruth" in d.name), None)
    if video_dir is None or gt_dir is None:
        raise SystemExit(f"Could not find the video and ground-truth folders under {root}")

    videos = {}
    for v in video_dir.rglob("*"):
        if v.suffix.lower() in VIDEO_EXTS:
            videos[(v.parent.name, v.stem)] = v
    pairs = []
    for x in sorted(gt_dir.rglob("*.gt.xml")):
        bg = x.parent.name.removesuffix("_gt")
        doc = x.name.removesuffix(".gt.xml")
        v = videos.get((bg, doc))
        if v is None:
            print(f"SKIP no video found for {bg}/{doc}")
            continue
        pairs.append((bg, doc, v, x))
    return pairs


def evaluate(pairs, step: int, tol: float, max_frames: int | None, out_dir: Path,
             strategy: str = "largest") -> pd.DataFrame:
    rows, saved = [], {}
    for bg, doc, video, xml in pairs:
        gt = load_gt(xml)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"SKIP cannot open video (codec?): {video}")
            continue
        n_done, idx = 0, 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            idx += 1                               # GT frame index is 1-based
            if (idx - 1) % step != 0 or idx not in gt:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            res = detect_page(frame, strategy=strategy)
            true = gt[idx]
            errs = corner_errors(res.corners, true)
            iou = quad_iou(res.corners, true)
            rows.append({
                "background": bg, "doc": doc, "frame": idx,
                "clean": bool(res.found), "method": res.method,
                "iou": round(iou, 4),
                "mean_corner_err_px": round(float(errs.mean()), 1),
                "max_corner_err_px": round(float(errs.max()), 1),
                "correct": corners_correct(res.corners, true, frame.shape, tol),
            })
            saved[(bg, doc, idx)] = (iou, frame, res.corners, true)
            n_done += 1
            if max_frames and n_done >= max_frames:
                break
        cap.release()
        print(f"{bg}/{doc}: evaluated {n_done} frame(s)")

    df = pd.DataFrame(rows)
    save_examples(df, saved, out_dir)
    return df


def draw(frame, found, true):
    img = frame.copy()
    t = max(2, img.shape[1] // 400)
    cv2.polylines(img, [true.astype(np.int32)], True, (0, 200, 0), t)       # green = true page
    cv2.polylines(img, [found.astype(np.int32)], True, (0, 0, 255), t)      # red = found page
    return img


def save_examples(df, saved, out_dir: Path, n: int = 6) -> None:
    """Save overlays of the worst and best frames (green = true, red = found)."""
    if df.empty:
        return
    (out_dir / "failures").mkdir(parents=True, exist_ok=True)
    (out_dir / "successes").mkdir(parents=True, exist_ok=True)
    ranked = df.sort_values("iou")
    for label, part in (("failures", ranked.head(n)), ("successes", ranked.tail(n))):
        for _, r in part.iterrows():
            iou, frame, found, true = saved[(r["background"], r["doc"], int(r["frame"]))]
            name = f"iou{iou:.2f}_{r['background']}_{r['doc']}_f{int(r['frame'])}.jpg"
            cv2.imwrite(str(out_dir / label / name), draw(frame, found, true))


def summarise(df: pd.DataFrame, out_dir: Path, tol: float) -> None:
    n = len(df)
    print(f"\n===== SmartDoc results ({n} frames) =====")
    print(f"Corners correct (all 4 within {tol:.0%} of image diagonal): {df['correct'].mean():.1%}")
    print(f"Clean 4-point contour found by the detector:               {df['clean'].mean():.1%}")
    print(f"Mean IoU with the true page:                               {df['iou'].mean():.3f}")
    print(f"Frames with IoU >= 0.90:                                   {(df['iou'] >= 0.90).mean():.1%}")
    print(f"Mean corner error: {df['mean_corner_err_px'].mean():.1f} px   "
          f"(median {df['mean_corner_err_px'].median():.1f} px)")

    by_bg = df.groupby("background").agg(frames=("iou", "size"), correct=("correct", "mean"),
                                         mean_iou=("iou", "mean")).round(3)
    by_doc = df.groupby("doc").agg(frames=("iou", "size"), correct=("correct", "mean"),
                                   mean_iou=("iou", "mean")).round(3)
    print("\nBy background:\n", by_bg)
    print("\nBy document:\n", by_doc)
    by_bg.to_csv(out_dir / "summary_by_background.csv")
    by_doc.to_csv(out_dir / "summary_by_document.csv")
    overall = pd.DataFrame([{
        "frames": n, "tolerance": tol, "pct_correct": round(df["correct"].mean() * 100, 1),
        "pct_clean": round(df["clean"].mean() * 100, 1), "mean_iou": round(df["iou"].mean(), 4),
        "pct_iou_ge_0.90": round((df["iou"] >= 0.90).mean() * 100, 1)}])
    overall.to_csv(out_dir / "summary_overall.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure page-detection accuracy on SmartDoc.")
    ap.add_argument("--root", type=Path, default=Path("data/smartdoc/sampleDataset/sampleDataset"))
    ap.add_argument("--video-dir", type=Path, default=None, help="override the video folder")
    ap.add_argument("--gt-dir", type=Path, default=None, help="override the ground-truth folder")
    ap.add_argument("--step", type=int, default=10, help="evaluate every Nth frame")
    ap.add_argument("--max-frames", type=int, default=None, help="cap frames per video")
    ap.add_argument("--tol", type=float, default=0.02, help="corner tolerance as a fraction of image diagonal")
    ap.add_argument("--strategy", choices=["largest", "robust"], default="largest",
                    help="page detection mode (see src/detect.py)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder (default results/smartdoc, or results/smartdoc_robust)")
    args = ap.parse_args()
    if args.out is None:
        args.out = Path("results/smartdoc" + ("_robust" if args.strategy == "robust" else ""))

    if not args.root.exists():
        raise SystemExit(f"Folder not found: {args.root}")
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(args.root, args.video_dir, args.gt_dir)
    if not pairs:
        raise SystemExit("No video + ground-truth pairs found.")
    df = evaluate(pairs, args.step, args.tol, args.max_frames, args.out, args.strategy)
    if df.empty:
        raise SystemExit("No frames were evaluated (could the videos be opened?).")
    df.to_csv(args.out / "frame_results.csv", index=False)
    summarise(df, args.out, args.tol)
    print(f"\nSaved tables to {args.out}\\ and overlays to {args.out}\\failures and \\successes")


if __name__ == "__main__":
    main()