# Smart Document Scanner and Enhancer

U21AMP05 Computer Vision | Assessment II | Project 1

Turns a phone photo of a page into a flat, clean scan, then measures how much
each enhancement method improves Tesseract OCR accuracy (character error rate).

## Pipeline

1. Grayscale, blur, Canny edges
2. Largest 4-point contour, then perspective transform
3. Enhancement: global threshold, adaptive threshold, CLAHE, shadow removal
4. OCR before/after, CER against typed ground truth
5. CLI / Streamlit app

## Folder structure

```
docscanner/
├── data/
│   ├── raw/             # own photos (not committed)
│   ├── smartdoc/        # SmartDoc dataset (not committed)
│   ├── ground_truth/    # typed .txt per OCR page (committed)
│   └── samples/         # 3-5 small sample images (committed)
├── src/
│   ├── detect.py        # phase 2: page detection
│   ├── warp.py          # phase 3: perspective correction
│   ├── enhance.py       # phase 4: enhancement + shadow removal
│   ├── ocr_eval.py      # phase 5: Tesseract + CER
│   └── pipeline.py      # glues the stages together
├── app/
│   ├── cli.py           # phase 6: command-line tool
│   └── streamlit_app.py
├── notebooks/           # experiments
├── results/             # figures, tables, CSVs (generated)
├── docs/
│   └── DATA_CHECKLIST.md
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tesseract itself must be installed separately:

- Windows: install from the UB Mannheim build, then add it to PATH
- Ubuntu: `sudo apt install tesseract-ocr`
- Colab: `!apt install tesseract-ocr`

Check with `tesseract --version`.

## How to run

Run everything from the project root with the virtual environment active.

**Scan one photo or a whole folder (command line)**

```bash
python -m app.cli data/raw/photo.jpg                 # -> results/scans/photo_scan.png
python -m app.cli data/raw --out results/scans        # every image in a folder
python -m app.cli photo.jpg -m adaptive               # pick the enhancement method
python -m app.cli photo.jpg --all --debug             # every method, a comparison grid, the page outline
python -m app.cli photo.jpg --ocr                     # also print the text Tesseract reads
```

Methods: `original`, `global`, `adaptive`, `clahe`, `shadow`, `shadow_otsu`.

**Web page (Streamlit)**

```bash
streamlit run app/streamlit_app.py
```

**Run the OCR experiment (CER table)**

1. Put photos in `data/raw/` and the typed text for each page in `data/ground_truth/`.
2. List them in `data/ground_truth/mapping.csv` (`image,ground_truth,condition,doc_type`).
3. `python -m src.ocr_eval` writes tables to `results/ocr/`.

**Individual stages**

```bash
python -m src.detect photo.jpg      # page outline
python -m src.warp photo.jpg        # flattened page
python -m src.enhance flat.png      # all enhancement methods + grid
```

## Random seed

Any random operation uses seed `42`.

## Team

| Roll No | Responsibility |
|---|---|
| 24CS002 | Pipeline Integration & UI Development (CLI `app/cli.py` & Streamlit `app/streamlit_app.py`) |
| 24CS003 | Perspective Transformation & Homography Warping (`src/warp.py` & `src/eval_smartdoc.py`) |
| 24CS008 | Image Preprocessing, Canny Edge Detection & Contour Optimization (`src/detect.py` & `src/eval_own_corners.py`) |
| 24CS017 | Illumination Enhancement, Shadow Removal & OCR CER Benchmarking (`src/enhance.py`, `src/ocr_eval.py` & results) |

## Data sources

- Self-captured photos (see `docs/DATA_CHECKLIST.md`)
- SmartDoc, ICDAR 2015 challenge (cite the paper in the report)