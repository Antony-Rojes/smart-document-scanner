"""Phase 3: perspective correction and corner-accuracy evaluation.

Takes the four page corners from detect.py and warps the photo into a flat,
top-down view of the page.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.detect import detect_page, order_points


def output_size(corners: np.ndarray) -> tuple[int, int]:
    """Width and height of the flattened page, from the longer of each pair of sides."""
    tl, tr, br, bl = corners
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    return int(round(width)), int(round(height))


def warp_page(image: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Warp the quadrilateral given by corners (TL, TR, BR, BL) to a rectangle.

    Returns the warped image and the 3x3 transform matrix.
    """
    corners = order_points(corners)
    w, h = output_size(corners)
    if w < 10 or h < 10:
        raise ValueError(f"Degenerate corners, output size would be {w}x{h}")
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, matrix, (w, h), flags=cv2.INTER_CUBIC)
    return warped, matrix


# ---------------------------------------------------------------------------
# Corner accuracy (used with SmartDoc ground-truth corners)
# ---------------------------------------------------------------------------

def corner_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Pixel distance between each predicted corner and its ground-truth corner."""
    return np.linalg.norm(order_points(pred) - order_points(gt), axis=1)


def quad_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-union of two convex quadrilaterals."""
    p = order_points(pred).astype(np.float32)
    g = order_points(gt).astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(p, g)
    union = cv2.contourArea(p) + cv2.contourArea(g) - inter
    return float(inter / union) if union > 0 else 0.0


def corners_correct(pred: np.ndarray, gt: np.ndarray, image_shape: tuple,
                    tol_fraction: float = 0.02) -> bool:
    """True if every corner is within tol_fraction of the image diagonal.

    Default 2% of the diagonal (about 40 px on a 2000 px diagonal). State the
    tolerance you used in the report.
    """
    h, w = image_shape[:2]
    tol = tol_fraction * float(np.hypot(h, w))
    return bool(np.all(corner_errors(pred, gt) <= tol))


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect the page and flatten it.")
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/warp"))
    args = ap.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"Could not read {args.image}")
    res = detect_page(img)
    print(f"detect: found={res.found} method={res.method}")
    if res.method == "fallback_full_image":
        print("No page found; warping the whole image, which does nothing useful.")
    warped, _ = warp_page(img, res.corners)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{args.image.stem}_warped.jpg"
    cv2.imwrite(str(out), warped)
    print(f"saved {out}  size={warped.shape[1]}x{warped.shape[0]}")


if __name__ == "__main__":
    main()