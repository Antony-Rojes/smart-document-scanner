"""Phase 4: enhancement and shadow removal.

Input: a flattened page (BGR or gray) from warp.py.
Output: cleaned grayscale images, one per method.

Methods
  global_threshold    Otsu's single threshold for the whole page
  adaptive_threshold  threshold computed per neighbourhood (handles uneven light)
  clahe               local contrast boost, keeps grayscale tones
  remove_shadow       estimate the page background, then divide it out
  shadow_then_otsu    shadow removal followed by Otsu (usually the strongest)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def global_threshold(image: np.ndarray) -> np.ndarray:
    """Otsu threshold on a lightly blurred copy. Output is 0/255."""
    gray = cv2.GaussianBlur(to_gray(image), (5, 5), 0)
    _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def adaptive_threshold(image: np.ndarray, block_size: int = 31, c: int = 15) -> np.ndarray:
    """Gaussian adaptive threshold. block_size must be odd.

    Larger block_size copes with larger shadows but blurs thin strokes.
    Larger c removes more faint background noise but may erase light ink.
    """
    if block_size % 2 == 0:
        block_size += 1
    gray = cv2.GaussianBlur(to_gray(image), (3, 3), 0)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )


def clahe(image: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization. Output stays grayscale."""
    op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return op.apply(to_gray(image))


def estimate_background(gray: np.ndarray, kernel_size: int = 0) -> np.ndarray:
    """Estimate the page background by morphological closing.

    Closing (dilate then erode) removes dark text strokes thinner than the
    kernel, leaving only the lighting. The kernel must be bigger than the
    thickest stroke; by default it scales with the page (about 3% of width).
    """
    if kernel_size <= 0:
        kernel_size = max(15, int(gray.shape[1] * 0.03)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return cv2.medianBlur(bg, 21 if kernel_size >= 21 else 11)


def remove_shadow(image: np.ndarray, kernel_size: int = 0) -> np.ndarray:
    """Divide the page by its estimated background. Returns normalised grayscale."""
    gray = to_gray(image)
    bg = estimate_background(gray, kernel_size)
    out = cv2.divide(gray, bg, scale=255)
    return cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)


def shadow_then_otsu(image: np.ndarray, kernel_size: int = 0) -> np.ndarray:
    return global_threshold(remove_shadow(image, kernel_size))


METHODS = {
    "original": to_gray,
    "global": global_threshold,
    "adaptive": adaptive_threshold,
    "clahe": clahe,
    "shadow": remove_shadow,
    "shadow_otsu": shadow_then_otsu,
}


def apply_all(image: np.ndarray) -> dict[str, np.ndarray]:
    """Run every method; returns {name: image}. Used for the before/after grid and OCR."""
    return {name: fn(image) for name, fn in METHODS.items()}


def make_grid(results: dict[str, np.ndarray], cell_width: int = 400) -> np.ndarray:
    """Side-by-side comparison with labels (for the report's before/after grid)."""
    cells = []
    for name, img in results.items():
        h = int(img.shape[0] * cell_width / img.shape[1])
        c = cv2.cvtColor(cv2.resize(img, (cell_width, h)), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(c, (0, 0), (cell_width, 28), (255, 255, 255), -1)
        cv2.putText(c, name, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cells.append(c)
    hmax = max(c.shape[0] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, hmax - c.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(128, 128, 128))
             for c in cells]
    return np.hstack(cells)


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply all enhancement methods to an image.")
    ap.add_argument("image", type=Path, help="a flattened page (output of warp.py)")
    ap.add_argument("--out", type=Path, default=Path("results/enhance"))
    args = ap.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"Could not read {args.image}")
    results = apply_all(img)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, out in results.items():
        cv2.imwrite(str(args.out / f"{args.image.stem}_{name}.png"), out)
    cv2.imwrite(str(args.out / f"{args.image.stem}_grid.jpg"), make_grid(results))
    print(f"saved {len(results)} images and a grid to {args.out}")


if __name__ == "__main__":
    main()