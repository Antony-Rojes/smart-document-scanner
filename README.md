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

_To be filled in as phases are completed._

## Random seed

Any random operation uses seed `42`.

## Team

| Roll No | Responsibility |
|---|---|
| 24CS002 | |
| 24CS003 | |
| 24CS008 | |
| 24CS017 | |

## Data sources

- Self-captured photos (see `docs/DATA_CHECKLIST.md`)
- SmartDoc, ICDAR 2015 challenge (cite the paper in the report)
