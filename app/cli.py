"""Phase 6: command-line tool.

Examples
  python -m app.cli photo.jpg                      # -> results/scans/photo_scan.png
  python -m app.cli photo.jpg -m adaptive          # choose the enhancement
  python -m app.cli data/raw --out results/scans   # a whole folder
  python -m app.cli photo.jpg --all                # save every method + a comparison grid
  python -m app.cli photo.jpg --ocr                # also print the text Tesseract reads
  python -m app.cli photo.jpg --debug              # also save the page outline overlay
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.enhance import METHODS, make_grid
from src.pipeline import DEFAULT_METHOD, scan

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_inputs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [path]


def load_ocr():
    """Import OCR lazily so the scanner works even when Tesseract is not installed."""
    try:
        from src.ocr_eval import run_ocr
        return run_ocr
    except SystemExit as e:  # raised by ocr_eval when tesseract.exe is not found
        print(f"OCR unavailable: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Turn a phone photo of a page into a clean flat scan.")
    ap.add_argument("input", type=Path, help="an image file, or a folder of images")
    ap.add_argument("--out", type=Path, default=Path("results/scans"), help="output folder")
    ap.add_argument("-m", "--method", default=DEFAULT_METHOD, choices=list(METHODS),
                    help=f"enhancement method (default: {DEFAULT_METHOD})")
    ap.add_argument("--all", action="store_true", help="save every method and a comparison grid")
    ap.add_argument("--upscale", type=float, default=1.0, help="enlarge the flattened page before enhancing")
    ap.add_argument("--strategy", choices=["largest", "robust"], default="largest", help="page detection mode")
    ap.add_argument("--ocr", action="store_true", help="print Tesseract's reading of the result")
    ap.add_argument("--debug", action="store_true", help="save the detected page outline")
    args = ap.parse_args()

    files = collect_inputs(args.input)
    if not files:
        raise SystemExit(f"No images found in {args.input}")
    args.out.mkdir(parents=True, exist_ok=True)
    run_ocr = load_ocr() if args.ocr else None

    ok = 0
    for f in files:
        photo = cv2.imread(str(f))
        if photo is None:
            print(f"SKIP  {f.name}: could not read image")
            continue
        result = scan(photo, upscale=args.upscale, strategy=args.strategy)
        status = "page found" if result.found else f"WARNING page not cleanly found ({result.detection.method})"
        out_path = args.out / f"{f.stem}_scan.png"
        cv2.imwrite(str(out_path), result.output(args.method))
        print(f"OK    {f.name}: {status} -> {out_path}")
        ok += 1

        if args.all:
            for name, img in result.enhanced.items():
                cv2.imwrite(str(args.out / f"{f.stem}_{name}.png"), img)
            cv2.imwrite(str(args.out / f"{f.stem}_grid.jpg"), make_grid(result.enhanced))
        if args.debug:
            cv2.imwrite(str(args.out / f"{f.stem}_outline.jpg"), result.detection.debug["overlay"])
        if run_ocr:
            print("----- OCR -----")
            print(run_ocr(result.output(args.method)).strip())
            print("---------------")

    print(f"\nDone: {ok}/{len(files)} image(s) scanned. Output folder: {args.out}")


if __name__ == "__main__":
    main()