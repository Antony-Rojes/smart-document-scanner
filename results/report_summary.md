# Assessment II: Experimental Results and Evaluation Summary

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