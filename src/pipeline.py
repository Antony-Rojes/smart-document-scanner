"""Glue: photo in, clean scan out.

    photo -> detect page -> flatten (perspective) -> enhance

Used by the command-line tool, the Streamlit app and the notebooks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.detect import DetectionResult, detect_page
from src.enhance import METHODS, apply_all
from src.warp import warp_page

# Default output method. "shadow" gives a clean grayscale page with even
# lighting. Change this to whichever method wins in your OCR results.
DEFAULT_METHOD = "shadow"


@dataclass
class ScanResult:
    detection: DetectionResult
    warped: np.ndarray                     # flattened page, colour
    enhanced: dict[str, np.ndarray] = field(default_factory=dict)  # method -> image

    @property
    def found(self) -> bool:
        return self.detection.found

    def output(self, method: str = DEFAULT_METHOD) -> np.ndarray:
        return self.enhanced[method]


def scan(photo: np.ndarray, upscale: float = 1.0, methods: list[str] | None = None,
         strategy: str = "largest") -> ScanResult:
    """Run the whole pipeline on one BGR photo.

    upscale  resize factor applied after flattening, before enhancement. Values
             above 1 can help OCR on small text (1.0 = off).
    methods  which enhancement methods to compute (default: all).
    strategy page detection mode: "largest" (textbook) or "robust" (scored candidates).
    """
    if photo is None or photo.size == 0:
        raise ValueError("Empty image")
    detection = detect_page(photo, strategy=strategy)
    warped, _ = warp_page(photo, detection.corners)
    if upscale and upscale != 1.0:
        warped = cv2.resize(warped, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

    if methods is None:
        enhanced = apply_all(warped)
    else:
        unknown = set(methods) - set(METHODS)
        if unknown:
            raise ValueError(f"Unknown method(s): {sorted(unknown)}. Choose from {list(METHODS)}")
        enhanced = {m: METHODS[m](warped) for m in methods}
    return ScanResult(detection=detection, warped=warped, enhanced=enhanced)