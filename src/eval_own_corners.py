"""Evaluate page corner detection accuracy on own photos.

Features:
1. Interactive labeling mode:
   python -m src.eval_own_corners --annotate
   Click TL, TR, BR, BL on each photo. Saves to data/ground_truth/corners_gt.json.

2. Benchmark mode:
   python -m src.eval_own_corners --strategy robust
   python -m src.eval_own_corners --strategy largest
   Compares detected corners against ground truth and reports:
   - % of images where all 4 corners are correct (within 2% diagonal tolerance)
   - Mean IoU (quadrilateral intersection over union)
   - Mean / median corner error in pixels
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.detect import detect_page, order_points
from src.warp import corner_errors, corners_correct, quad_iou

GT_FILE = Path("data/ground_truth/corners_gt.json")
RAW_DIR = Path("data/raw")


def annotate_photos(raw_dir: Path, gt_path: Path):
    gt = {}
    if gt_path.exists():
        try:
            gt = json.loads(gt_path.read_text(encoding="utf-8"))
        except Exception:
            gt = {}

    images = sorted(list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png")))
    print(f"Starting corner annotation for {len(images)} images.")
    print("Click 4 corners in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.")
    print("Press 'r' to reset points on current image, 's' to skip, 'q' to quit.")

    for img_path in images:
        name = img_path.name
        if name in gt:
            print(f"Skipping {name} (already annotated). Delete from JSON to re-annotate.")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        disp_h = 800
        scale = disp_h / float(h)
        disp_w = int(round(w * scale))
        disp = cv2.resize(img, (disp_w, disp_h))

        clicks = []

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(clicks) < 4:
                    clicks.append((x / scale, y / scale))
                    cv2.circle(disp, (x, y), 5, (0, 0, 255), -1)
                    labels = ["TL", "TR", "BR", "BL"]
                    cv2.putText(disp, labels[len(clicks) - 1], (x + 8, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Annotate Corners", disp)

        cv2.namedWindow("Annotate Corners")
        cv2.setMouseCallback("Annotate Corners", on_mouse)

        while True:
            cv2.imshow("Annotate Corners", disp)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                gt_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")
                return
            elif key == ord("s"):
                break
            elif key == ord("r"):
                clicks = []
                disp = cv2.resize(img, (disp_w, disp_h))
            elif len(clicks) == 4:
                gt[name] = order_points(np.array(clicks, dtype=np.float32)).tolist()
                print(f"Saved corners for {name}")
                break

        cv2.destroyAllWindows()

    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")
    print(f"Annotation complete. Saved to {gt_path}")


def evaluate_own_corners(raw_dir: Path, gt_path: Path, strategy: str = "robust",
                         tol_fraction: float = 0.02, out_dir: Path = Path("results/own_corners")) -> pd.DataFrame:
    if not gt_path.exists():
        raise SystemExit(f"Ground truth file {gt_path} not found. Run with --annotate first.")

    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for name, pts in gt.items():
        img_path = raw_dir / name
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gt_corners = np.array(pts, dtype=np.float32)
        res = detect_page(img, strategy=strategy)

        diag = float(np.hypot(img.shape[0], img.shape[1]))
        errs = corner_errors(res.corners, gt_corners)
        mean_err = float(errs.mean())
        max_err = float(errs.max())
        iou = quad_iou(res.corners, gt_corners)
        correct = corners_correct(res.corners, gt_corners, img.shape, tol_fraction=tol_fraction)

        records.append({
            "image": name,
            "strategy": strategy,
            "found": res.found,
            "method": res.method,
            "mean_corner_err_px": round(mean_err, 2),
            "max_corner_err_px": round(max_err, 2),
            "corner_err_pct_diag": round((mean_err / diag) * 100, 2),
            "iou": round(iou, 4),
            "correct": correct,
        })

    df = pd.DataFrame(records)
    csv_path = out_dir / f"own_corners_{strategy}.csv"
    df.to_csv(csv_path, index=False)

    pct_correct = (df["correct"].mean() * 100) if not df.empty else 0.0
    mean_iou = df["iou"].mean() if not df.empty else 0.0
    mean_err = df["mean_corner_err_px"].mean() if not df.empty else 0.0
    median_err = df["mean_corner_err_px"].median() if not df.empty else 0.0

    print(f"\n===== Own Photos Corner Evaluation ({strategy}) =====")
    print(f"Total evaluated photos: {len(df)}")
    print(f"Corners correct (<{tol_fraction*100:.1f}% diagonal): {pct_correct:.1f}%")
    print(f"Mean IoU with ground truth:       {mean_iou:.4f}")
    print(f"Mean corner error:                 {mean_err:.1f} px (median {median_err:.1f} px)")
    print(f"Saved results to {csv_path}\n")

    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate corner detection on own photos.")
    parser.add_argument("--annotate", action="store_true", help="Launch interactive corner annotator")
    parser.add_argument("--strategy", choices=["largest", "robust"], default="robust")
    parser.add_argument("--tol", type=float, default=0.02, help="Tolerance fraction of diagonal")
    parser.add_argument("--gt", type=Path, default=GT_FILE)
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=Path("results/own_corners"))
    args = parser.parse_args()

    if args.annotate:
        annotate_photos(args.raw, args.gt)
    else:
        evaluate_own_corners(args.raw, args.gt, strategy=args.strategy, tol_fraction=args.tol, out_dir=args.out)


if __name__ == "__main__":
    main()
