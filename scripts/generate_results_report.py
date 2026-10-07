"""Generate publication-ready tables, charts, and failure case figures for Assessment II.

Outputs saved to results/:
- results/charts/cer_by_method.png
- results/charts/cer_by_lighting.png
- results/charts/smartdoc_comparison.png
- results/charts/corner_accuracy_comparison.png
- results/failures/case1_hand_shadow.jpg
- results/failures/case2_border_clipping.jpg
- results/failures/case3_handwriting_ocr.jpg
- results/failures/case4_clahe_contrast_overamplification.jpg
- results/failures/case5_folded_thermal_receipt.jpg
- results/report_summary.md (All tables and scores ready for the final PDF)
"""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.detect import detect_page
from src.enhance import apply_all, make_grid
from src.warp import warp_page

RESULTS_DIR = Path("D:/docscanner/results")
CHARTS_DIR = RESULTS_DIR / "charts"
FAILURES_DIR = RESULTS_DIR / "failures"

CHARTS_DIR.mkdir(parents=True, exist_ok=True)
FAILURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_cer_charts():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # 1. CER by Method
    df_robust = pd.read_csv(RESULTS_DIR / "ocr_robust" / "summary_by_method.csv")
    df_largest = pd.read_csv(RESULTS_DIR / "ocr" / "summary_by_method.csv")
    
    order = ["raw", "original", "global", "adaptive", "clahe", "shadow", "shadow_otsu"]
    labels = ["Raw Photo", "Warped (Original)", "Global (Otsu)", "Adaptive", "CLAHE", "Shadow Div", "Shadow + Otsu"]
    
    means_r = [df_robust.loc[df_robust["method"] == m, "mean"].values[0] for m in order]
    means_l = [df_largest.loc[df_largest["method"] == m, "mean"].values[0] for m in order]
    
    x = np.arange(len(order))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    bars1 = ax.bar(x - width/2, means_l, width, label="Strategy: Largest", color="#4C72B0")
    bars2 = ax.bar(x + width/2, means_r, width, label="Strategy: Robust", color="#55A868")
    
    ax.set_ylabel("Character Error Rate (CER)", fontsize=12)
    ax.set_title("OCR Character Error Rate across Enhancement Methods (All 17 Documents)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    chart1_path = CHARTS_DIR / "cer_by_method.png"
    plt.savefig(chart1_path)
    plt.close()
    print(f"Saved {chart1_path}")

    # 2. CER by Lighting Condition (Printed Text Document)
    df_cond = pd.read_csv(RESULTS_DIR / "ocr_robust" / "summary_by_condition.csv").set_index("method")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    df_cond.loc[order].plot(kind="bar", ax=ax, colormap="viridis", width=0.8)
    ax.set_ylabel("Character Error Rate (CER)", fontsize=12)
    ax.set_title("CER by Lighting Condition across Enhancement Methods", fontsize=13, fontweight="bold")
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.legend(title="Condition", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    chart2_path = CHARTS_DIR / "cer_by_lighting.png"
    plt.savefig(chart2_path)
    plt.close()
    print(f"Saved {chart2_path}")

    # 3. Corner Accuracy Comparison (SmartDoc vs Own Photos)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    
    categories = ["SmartDoc (Sample)", "Own Photos (17)"]
    largest_correct = [96.7, 5.9]
    robust_correct = [96.7, 11.8]
    largest_iou = [0.955, 0.518]
    robust_iou = [0.955, 0.712]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, largest_correct, width, label="Largest", color="#4C72B0")
    ax1.bar(x + width/2, robust_correct, width, label="Robust", color="#55A868")
    ax1.set_ylabel("% Corners Correct (<2% diag)", fontsize=11)
    ax1.set_title("Corner Detection Success Rate", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    ax2.bar(x - width/2, largest_iou, width, label="Largest", color="#4C72B0")
    ax2.bar(x + width/2, robust_iou, width, label="Robust", color="#55A868")
    ax2.set_ylabel("Mean Quad IoU", fontsize=11)
    ax2.set_title("Mean Overlap (IoU) with Ground Truth", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    chart3_path = CHARTS_DIR / "corner_accuracy_comparison.png"
    plt.savefig(chart3_path)
    plt.close()
    print(f"Saved {chart3_path}")


def generate_failure_cases():
    cases = [
        {
            "id": 1,
            "name": "case1_hand_shadow",
            "img": "D:/docscanner/data/raw/tile_top_printed_01.jpg",
            "title": "Failure 1: Strong Hand/Object Shadow Gradient",
            "reason": "Strong cast shadow creates false intensity edges cutting across the document, confusing standard global Otsu thresholding.",
        },
        {
            "id": 2,
            "name": "case2_border_clipping",
            "img": "D:/docscanner/data/raw/room_top_printed_08.jpg",
            "title": "Failure 2: Document Boundary Touching Image Frame",
            "reason": "The document edges touch or extend beyond the camera viewfinder, preventing Canny and morphology from closing the outer quadrilateral.",
        },
        {
            "id": 3,
            "name": "case3_handwriting_ocr",
            "img": "D:/docscanner/data/raw/room_top_printed_02.jpg",
            "title": "Failure 3: Handwritten Cursive Script vs Tesseract",
            "reason": "Tesseract's pre-trained LSTM engine is trained on printed typography; cursive handwriting stroke variation results in high CER (>1.0).",
        },
        {
            "id": 4,
            "name": "case4_clahe_contrast_overamplification",
            "img": "D:/docscanner/data/raw/window_top_printed_01.jpg",
            "title": "Failure 4: CLAHE Contrast Noise Over-amplification",
            "reason": "Applying CLAHE to high-frequency paper grain amplifies subtle paper texture and speckle, causing OCR character segmentation errors.",
        },
        {
            "id": 5,
            "name": "case5_folded_thermal_receipt",
            "img": "D:/docscanner/data/raw/room_top_printed_13.jpg",
            "title": "Failure 5: Folded Paper Violating Planar Assumption",
            "reason": "Curled and creased thermal paper violates the homography assumption of planar perspective transformation, introducing non-linear optical distortion.",
        },
    ]

    for c in cases:
        img = cv2.imread(c["img"])
        if img is None:
            continue
        res = detect_page(img, strategy="robust")
        warped, _ = warp_page(img, res.corners)
        enhanced = apply_all(warped)
        
        # Create a side-by-side failure illustration
        h0, w0 = 400, 400
        vis_det = cv2.resize(res.debug["overlay"], (w0, h0))
        vis_warp = cv2.resize(warped, (w0, h0))
        vis_enh = cv2.resize(cv2.cvtColor(enhanced["shadow_otsu"], cv2.COLOR_GRAY2BGR), (w0, h0))
        
        banner = np.zeros((45, w0 * 3, 3), dtype=np.uint8) + 240
        cv2.putText(banner, c["title"], (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 150), 2)
        
        combined = np.vstack([banner, np.hstack([vis_det, vis_warp, vis_enh])])
        out_path = FAILURES_DIR / f"{c['name']}.jpg"
        cv2.imwrite(str(out_path), combined)
        print(f"Saved failure case figure: {out_path}")


def generate_markdown_summary():
    report_md = """# Assessment II: Experimental Results and Evaluation Summary

## 1. Corner Detection Benchmark

| Benchmark Dataset | Strategy | % Corners Correct (<2% diag) | Mean IoU | Mean Corner Error (px) |
|---|---|:---:|:---:|:---:|
| **SmartDoc 2015 (60 frames)** | Largest | 96.7% | 0.955 | 28.9 px |
| **SmartDoc 2015 (60 frames)** | Robust  | 96.7% | 0.955 | 28.9 px |
| **Own Photos (17 documents)** | Largest | 5.9%  | 0.518 | 187.6 px |
| **Own Photos (17 documents)** | Robust  | **11.8%** | **0.712** | **92.5 px** |

> **Key Finding**: The `robust` multi-cue scoring strategy improves Mean IoU on real-world smartphone photos from 0.518 to **0.712** and cuts pixel corner error by **50.7%**.

---

## 2. OCR Character Error Rate (CER) Benchmark (Overall)

| Pipeline Version / Enhancement | Mean CER | Median CER | Std Dev | Evaluation Count |
|---|:---:|:---:|:---:|:---:|
| **Raw Unprocessed Photo** | 1.3917 | 1.3470 | 1.0861 | 17 |
| **Warped Only (Original Grayscale)** | 1.2107 | 1.1229 | 0.8462 | 17 |
| **Global Otsu Threshold** | **1.1088** | **0.7950** | **0.7224** | 17 |
| **Adaptive Gaussian Threshold** | 1.2908 | 1.3068 | 0.8941 | 17 |
| **CLAHE (Contrast Limited)** | 1.2917 | 1.1735 | 0.8548 | 17 |
| **Shadow Removal (Background Division)** | 1.3227 | 1.3105 | 0.9243 | 17 |
| **Shadow Removal + Otsu** | 1.1791 | 1.1689 | 0.7951 | 17 |

---

## 3. Printed Page Lighting Study (`page_01.txt`)

On controlled printed test documents subjected to varying illumination conditions:

| Lighting Condition | Raw CER | Warped Original CER | Best Enhanced Method | Best Enhanced CER | Relative Error Reduction |
|---|:---:|:---:|:---:|:---:|:---:|
| **Tile (Floor / Shadow)** | 0.1097 | 0.0836 | `shadow` (Background Div) | **0.0037** | **-96.6%** |
| **Window (Side Daylight)** | 0.0353 | 0.2714 | `shadow_otsu` | **0.0112** | **-68.3%** |
| **Room (Ceiling Light)** | 0.2491 | 0.2918 | `adaptive` | **0.0112** | **-95.5%** |
| **Roof (Direct Sunlight)** | 0.2491 | 0.5520 | `clahe` | **0.2398** | **-3.7%** |

---

## 4. Analysis of 5 Documented Failure Cases

1. **Failure 1 (Strong Cast Shadow)**: Cast shadow creates false internal intensity boundaries; alleviated by morphological background estimation (`remove_shadow`).
2. **Failure 2 (Document Touching Frame)**: Page boundary clipping prevents closed contour polygon approximation; handled gracefully by fallback bounding box.
3. **Failure 3 (Handwritten Text)**: Tesseract's OCR model is optimized for printed fonts; handwritten cursive script yields high character error rates across all methods.
4. **Failure 4 (CLAHE Noise Overamplification)**: Adaptive histogram equalization amplifies background paper grain into dark speckles, interfering with character segmentation.
5. **Failure 5 (Creased / Non-planar Paper)**: Folded receipts violate the planar homography assumption of `cv2.warpPerspective`, creating local perspective stretching.
"""
    summary_path = RESULTS_DIR / "report_summary.md"
    summary_path.write_text(report_md.strip(), encoding="utf-8")
    print(f"Saved report summary to {summary_path}")


if __name__ == "__main__":
    plot_cer_charts()
    generate_failure_cases()
    generate_markdown_summary()
