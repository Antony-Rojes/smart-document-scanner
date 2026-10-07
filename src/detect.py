"""Phase 2: page detection.

Finds the four corners of a document page in a phone photo.
Supports two strategies:
  - "largest": standard Canny + contour hierarchy + largest convex 4-point approximation.
  - "robust": multi-cue candidate extraction (Canny, Otsu, min-channel) with scoring based on
              edge support, rectangularity, aspect ratio, border penalty, and contrast.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

DETECT_HEIGHT = 500
MIN_AREA_FRACTION = 0.05


@dataclass
class DetectionResult:
    corners: np.ndarray            # (4, 2) float32, TL, TR, BR, BL, original coords
    found: bool                    # True if a valid document page was found
    method: str                    # "approx" | "minAreaRect" | "robust" | "fallback_full_image"
    debug: dict = field(default_factory=dict)  # intermediate images


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]  # y - x
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def _approx_quad(contour):
    """Convex 4-point approximation of a contour (or its convex hull)."""
    for src in (contour, cv2.convexHull(contour)):
        peri = cv2.arcLength(src, True)
        for eps in (0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08):
            a = cv2.approxPolyDP(src, eps * peri, True)
            if len(a) == 4 and cv2.isContourConvex(a):
                return a.reshape(4, 2).astype(np.float32)
    return None


def _quad_candidates(small: np.ndarray):
    h, w = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    minch = cv2.GaussianBlur(small.min(axis=2), (5, 5), 0)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    med = float(np.median(blur))
    edge_masks = []
    for lo, hi in ((50, 150), (20, 60), (max(10, int(0.66 * med)), max(30, int(1.33 * med)))):
        e = cv2.Canny(blur, lo, hi)
        edge_masks.append(cv2.morphologyEx(e, cv2.MORPH_CLOSE, close_k, iterations=3))

    region_masks = []
    local_bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(3, int(0.12 * h)) | 1)
    diff = (blur.astype(np.int16) - local_bg.astype(np.int16))
    region_masks.append(((diff > 8).astype(np.uint8) * 255))
    for ch in (blur, minch):
        _, m = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(m) < 127:
            m = cv2.bitwise_not(m)
        region_masks.append(m)

    cands = []
    for m in edge_masks:
        cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
            if cv2.contourArea(c) < MIN_AREA_FRACTION * h * w:
                break
            q = _approx_quad(c)
            if q is not None:
                cands.append(q)

    for m in region_masks:
        cleaned = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, open_k), cv2.MORPH_CLOSE, open_k)
        cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
            if cv2.contourArea(c) < MIN_AREA_FRACTION * h * w:
                break
            q = _approx_quad(c)
            if q is not None:
                cands.append(q)

    return cands, gray, blur


def _score_quad(q: np.ndarray, gray: np.ndarray, edge_map: np.ndarray):
    """Return (score, edge_support, corners_near_border). Score 0 means rejected."""
    h, w = gray.shape
    q = order_points(q)
    af = cv2.contourArea(q) / float(h * w)
    if af < MIN_AREA_FRACTION or af > 0.98:
        return 0.0, 0.0, 0

    angles = []
    for i in range(4):
        v1 = q[i - 1] - q[i]
        v2 = q[(i + 1) % 4] - q[i]
        cosv = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
        angles.append(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))
    if min(angles) < 50 or max(angles) > 130:
        return 0.0, 0.0, 0
    rect = float(np.clip(1.0 - np.mean(np.abs(np.array(angles) - 90.0)) / 45.0, 0.0, 1.0))

    wd = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2.0
    ht = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2.0
    ratio = max(wd, ht) / max(1e-6, min(wd, ht))
    ar_pen = 1.0 if ratio <= 2.2 else max(0.3, 2.2 / ratio)

    t = np.linspace(0.05, 0.95, 30)
    pts = np.concatenate([q[i] + (q[(i + 1) % 4] - q[i]) * t[:, None] for i in range(4)])
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    support = float((edge_map[ys, xs] > 0).mean())

    margin = 0.015 * max(h, w)
    near = int(sum(x < margin or y < margin or x > w - 1 - margin or y > h - 1 - margin for x, y in q))
    # Do not penalize large documents that naturally fill the frame
    border_pen = 1.0 if (near == 0 or af > 0.45) else (0.75 if near == 1 else 0.4)

    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, q.astype(np.int32), 1)
    ring_w = (int(0.04 * h) | 1)
    ring = (cv2.dilate(mask, np.ones((ring_w, ring_w), np.uint8)) > 0) & (mask == 0)
    inside = float(gray[mask > 0].mean()) if mask.any() else 0.0
    outside = float(gray[ring].mean()) if ring.any() else inside
    contrast = inside - outside
    contrast_f = 0.8 if contrast < 0 else 1.0 + 0.5 * min(contrast / 50.0, 1.0)

    # Heavily favor large page quads over tiny inner paragraphs
    area_weight = af ** 0.85 if af >= 0.20 else (af ** 1.5) * 0.2
    score = (0.20 + support) * (0.4 + 0.6 * rect) * area_weight * ar_pen * border_pen * contrast_f
    return float(score), support, near


def _robust_quad(small: np.ndarray):
    """Best-scoring page quad as (corners, support, near_border) or None."""
    cands, gray, blur = _quad_candidates(small)
    if not cands:
        return None
    canny = cv2.Canny(blur, 20, 60)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    grad = cv2.morphologyEx(otsu, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    edge_map = cv2.dilate(cv2.bitwise_or(canny, grad), np.ones((3, 3), np.uint8))

    best, best_score = None, 0.0
    for q in cands:
        s, support, near = _score_quad(q, gray, edge_map)
        if s > best_score:
            best, best_score = (order_points(q), support, near), s
    return best


def detect_page(
    image: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    blur_ksize: int = 5,
    strategy: str = "robust",
) -> DetectionResult:
    """Detect the page in a BGR image and return its four corners.

    strategy: "largest" or "robust" (default: "robust")
    """
    if strategy not in ("largest", "robust"):
        raise ValueError("strategy must be 'largest' or 'robust'")

    h0, w0 = image.shape[:2]
    scale = DETECT_HEIGHT / float(h0)
    small = cv2.resize(image, (int(round(w0 * scale)), DETECT_HEIGHT), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

    if strategy == "robust":
        best = _robust_quad(small)
        if best is not None:
            corners_small, support, near = best
            corners = (corners_small / scale).astype(np.float32)
            found = bool(support >= 0.25 and near < 3)
            overlay = image.copy()
            cv2.polylines(overlay, [corners.astype(np.int32)], True, (0, 255, 0), max(3, w0 // 250))
            for i, (x, y) in enumerate(corners):
                cv2.circle(overlay, (int(x), int(y)), max(8, w0 // 120), (0, 0, 255), -1)
                cv2.putText(overlay, "TL TR BR BL".split()[i], (int(x) + 10, int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, max(0.8, w0 / 1500), (255, 0, 0), 2)
            return DetectionResult(
                corners=corners,
                found=found,
                method="robust",
                debug={"gray": gray, "blurred": blurred, "edges": edges,
                       "edges_closed": edges_closed, "overlay": overlay},
            )

    # Strategy "largest" fallback/primary
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_AREA_FRACTION * small.shape[0] * small.shape[1]
    
    quad = None
    biggest = None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for c in contours:
        if cv2.contourArea(c) < min_area:
            break
        q = _approx_quad(c)
        if q is not None:
            quad = q
            break
        if biggest is None:
            biggest = c

    if quad is not None:
        corners_small, found, method = order_points(quad), True, "approx"
    elif biggest is not None:
        corners_small = order_points(cv2.boxPoints(cv2.minAreaRect(biggest)))
        found, method = True, "minAreaRect"
    else:
        hs, ws = small.shape[:2]
        corners_small = np.array([[0, 0], [ws - 1, 0], [ws - 1, hs - 1], [0, hs - 1]], np.float32)
        found, method = False, "fallback_full_image"

    corners = (corners_small / scale).astype(np.float32)
    overlay = image.copy()
    cv2.polylines(overlay, [corners.astype(np.int32)], True, (0, 255, 0), max(3, w0 // 250))
    for i, (x, y) in enumerate(corners):
        cv2.circle(overlay, (int(x), int(y)), max(8, w0 // 120), (0, 0, 255), -1)
        cv2.putText(overlay, "TL TR BR BL".split()[i], (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.8, w0 / 1500), (255, 0, 0), 2)

    return DetectionResult(
        corners=corners,
        found=found,
        method=method,
        debug={"gray": gray, "blurred": blurred, "edges": edges,
               "edges_closed": edges_closed, "overlay": overlay},
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect the page corners in a photo.")
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/detect"))
    ap.add_argument("--strategy", choices=["largest", "robust"], default="robust")
    args = ap.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"Could not read {args.image}")
    res = detect_page(img, strategy=args.strategy)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    cv2.imwrite(str(args.out / f"{stem}_edges.png"), res.debug["edges_closed"])
    cv2.imwrite(str(args.out / f"{stem}_overlay.jpg"), res.debug["overlay"])
    print(f"found={res.found} method={res.method}")
    print(res.corners.round(1))


if __name__ == "__main__":
    main()
