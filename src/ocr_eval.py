"""Phase 5: OCR evaluation.

For every page in a mapping CSV this runs the full pipeline, then Tesseract on
several versions of the page, and scores each against the typed ground truth
with the character error rate (CER).

    CER = (substitutions + deletions + insertions) / number of reference characters

Versions compared per page
  raw          the unprocessed phone photo (no detection, no warp)
  original     page found and flattened, grayscale only (no enhancement)
  global       + Otsu threshold
  adaptive     + adaptive threshold
  clahe        + CLAHE
  shadow       + shadow removal (divide by background)
  shadow_otsu  + shadow removal then Otsu

"raw" vs "original" isolates what the warp alone does. "original" vs the rest
isolates what each enhancement adds.

mapping.csv columns (header required)
  image         path to the photo, relative to the CSV's folder or absolute
  ground_truth  path to the typed .txt
  condition     optional label, e.g. room / window / shadow
  doc_type      optional label, e.g. printed / notebook
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytesseract

from src.detect import detect_page
from src.enhance import apply_all, to_gray
from src.warp import warp_page


def _locate_tesseract() -> None:
    """Point pytesseract at tesseract.exe if it is not on PATH (common on Windows).

    Order: TESSERACT_CMD environment variable, PATH, then the default install folders.
    """
    import os
    import shutil

    candidates = [os.environ.get("TESSERACT_CMD"), shutil.which("tesseract"),
                  r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                  r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                  str(Path.home() / "AppData/Local/Programs/Tesseract-OCR/tesseract.exe")]
    for c in candidates:
        if c and Path(c).exists():
            pytesseract.pytesseract.tesseract_cmd = str(c)
            return
    raise SystemExit(
        "Tesseract is not installed (or not found).\n"
        "Windows: install the UB Mannheim build from https://github.com/UB-Mannheim/tesseract/wiki\n"
        "then re-run. Or set the TESSERACT_CMD environment variable to the full path of tesseract.exe."
    )


_locate_tesseract()

try:  # fast C implementation if installed
    import Levenshtein

    def edit_distance(a: str, b: str) -> int:
        return Levenshtein.distance(a, b)
except ImportError:  # pure-python fallback so the code always runs
    def edit_distance(a: str, b: str) -> int:
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def normalize(text: str, lowercase: bool = False) -> str:
    """Collapse all whitespace to single spaces. Case-sensitive unless asked."""
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if lowercase else text


def cer(reference: str, hypothesis: str, lowercase: bool = False) -> float:
    """Character error rate. Can exceed 1.0 if the hypothesis is much longer."""
    ref = normalize(reference, lowercase)
    hyp = normalize(hypothesis, lowercase)
    if not ref:
        raise ValueError("Empty reference text")
    return edit_distance(ref, hyp) / len(ref)


def run_ocr(image: np.ndarray, psm: int = 6, lang: str = "eng") -> str:
    """Tesseract on one image. psm 6 = a single uniform block of text."""
    return pytesseract.image_to_string(image, lang=lang, config=f"--psm {psm}")


def evaluate_page(photo: np.ndarray, truth: str, psm: int = 6, lowercase: bool = False,
                  strategy: str = "largest"):
    """Return (rows, info) where rows is {method: (cer, ocr_text)}."""
    det = detect_page(photo, strategy=strategy)
    warped, _ = warp_page(photo, det.corners)

    versions = {"raw": to_gray(photo)}
    versions.update(apply_all(warped))  # original, global, adaptive, clahe, shadow, shadow_otsu

    rows = {}
    for name, img in versions.items():
        text = run_ocr(img, psm=psm)
        rows[name] = (cer(truth, text, lowercase), text)
    return rows, {"corners_found": det.found, "detect_method": det.method}


def evaluate_mapping(mapping_csv: Path, out_dir: Path, psm: int = 6, lowercase: bool = False,
                     strategy: str = "largest") -> pd.DataFrame:
    base = mapping_csv.parent
    if not mapping_csv.exists():
        raise SystemExit(f"Mapping file not found: {mapping_csv}")
    try:
        df_map = pd.read_csv(mapping_csv)
    except pd.errors.EmptyDataError:
        raise SystemExit(
            f"{mapping_csv} is empty. It needs a header line and one row per page:\n"
            "  image,ground_truth,condition,doc_type\n"
            "  ../raw/room_top_printed_01.jpg,page_01.txt,room,printed"
        )
    missing = {"image", "ground_truth"} - set(df_map.columns)
    if missing:
        raise SystemExit(f"{mapping_csv} is missing column(s): {sorted(missing)}")
    if df_map.empty:
        raise SystemExit(f"{mapping_csv} has a header but no rows yet. Add one row per page.")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ocr_text").mkdir(exist_ok=True)

    records = []
    for _, row in df_map.iterrows():
        img_path = Path(row["image"])
        img_path = img_path if img_path.is_absolute() else base / img_path
        gt_path = Path(row["ground_truth"])
        gt_path = gt_path if gt_path.is_absolute() else base / gt_path

        photo = cv2.imread(str(img_path))
        if photo is None:
            print(f"SKIP unreadable image: {img_path}")
            continue
        if not gt_path.exists():
            print(f"SKIP no ground-truth text yet: {gt_path}  (create it with the exact page text)")
            continue
        truth = gt_path.read_text(encoding="utf-8")
        if not normalize(truth):
            print(f"SKIP ground-truth file is empty: {gt_path}")
            continue

        rows, info = evaluate_page(photo, truth, psm=psm, lowercase=lowercase, strategy=strategy)
        for method, (score, text) in rows.items():
            (out_dir / "ocr_text" / f"{img_path.stem}__{method}.txt").write_text(text, encoding="utf-8")
            records.append({
                "image": img_path.name,
                "condition": row.get("condition", ""),
                "doc_type": row.get("doc_type", ""),
                "method": method,
                "cer": score,
                **info,
            })
        best = min(rows, key=lambda m: rows[m][0])
        print(f"{img_path.name:32s} corners={info['detect_method']:20s} "
              f"original={rows['original'][0]:.3f}  best={best} ({rows[best][0]:.3f})")

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "ocr_results.csv", index=False)
    return df


ORDER = ["raw", "original", "global", "adaptive", "clahe", "shadow", "shadow_otsu"]


def summarise(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Mean/median CER per method, plus per condition. Saves CSVs for the report."""
    summary = (df.groupby("method")["cer"].agg(["mean", "median", "std", "count"])
               .reindex(ORDER).round(4))
    summary.to_csv(out_dir / "summary_by_method.csv")

    if df["condition"].astype(str).str.len().gt(0).any():
        by_cond = df.pivot_table(index="method", columns="condition", values="cer", aggfunc="mean").reindex(ORDER).round(4)
        by_cond.to_csv(out_dir / "summary_by_condition.csv")
        print("\nMean CER by condition:\n", by_cond)

    # Failure candidates: pages where even the best method is poor, or no clean corners.
    best = df.groupby("image")["cer"].min().rename("best_cer")
    corners = df.groupby("image")["corners_found"].first()
    fail = pd.concat([best, corners], axis=1)
    fail = fail[(fail["best_cer"] > 0.25) | (~fail["corners_found"].astype(bool))].sort_values("best_cer", ascending=False)
    fail.to_csv(out_dir / "failure_candidates.csv")
    print("\nMean CER by method (lower is better):\n", summary)
    print(f"\n{len(fail)} failure candidates saved to failure_candidates.csv")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="OCR before/after enhancement, scored by CER.")
    ap.add_argument("--mapping", type=Path, default=Path("data/ground_truth/mapping.csv"))
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder (default results/ocr, or results/ocr_robust)")
    ap.add_argument("--strategy", choices=["largest", "robust"], default="largest",
                    help="page detection mode (see src/detect.py)")
    ap.add_argument("--psm", type=int, default=6, help="Tesseract page segmentation mode")
    ap.add_argument("--lowercase", action="store_true", help="ignore case when scoring")
    args = ap.parse_args()

    if args.out is None:
        args.out = Path("results/ocr" + ("_robust" if args.strategy == "robust" else ""))
    df = evaluate_mapping(args.mapping, args.out, args.psm, args.lowercase, args.strategy)
    if df.empty:
        raise SystemExit("No pages evaluated. Check mapping.csv paths.")
    summarise(df, args.out)


if __name__ == "__main__":
    main()