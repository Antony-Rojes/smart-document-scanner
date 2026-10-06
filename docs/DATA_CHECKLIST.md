# Data Collection Checklist (Phase 1)

## A. Own photos: 60+ minimum

Capture matrix: **3 lighting x 4 angles x 5 document types = 60**.
Take 1-2 extra of each trouble case so you finish above 60.

**Lighting**
- [ ] Room light (tube light / ceiling)
- [ ] Window light (side daylight, some gradient across the page)
- [ ] Hand/phone shadow falling on the page

**Angle**
- [ ] Straight from above (near top-down)
- [ ] Tilted left/right (about 30 degrees)
- [ ] Tilted forward/back (about 30 degrees)
- [ ] Strong oblique (about 45 degrees)

**Document type**
- [ ] Ruled notebook page, handwritten
- [ ] Plain/blank page, handwritten
- [ ] Printed A4 text sheet
- [ ] Form or table
- [ ] Board (whiteboard/blackboard)

**Capture rules**
- Use the same phone for all photos, no filters, no flash for most (try a few with flash)
- Keep all four page corners inside the frame
- Vary the background: dark desk, light desk, cluttered desk
- Keep photos at original resolution; do not crop

**Naming convention:** `{light}_{angle}_{type}_{nn}.jpg`
Examples: `room_top_printed_01.jpg`, `window_obl_notebook_02.jpg`, `shadow_tiltL_form_01.jpg`

## B. OCR pages with ground truth: 15+ minimum

Tesseract is poor at handwriting, so build the OCR set mostly from
**printed pages** (print or use textbooks/notes). A few clean handwritten
pages are fine, but expect high CER and note it in the analysis.

- [ ] Pick 15+ pages (aim for 18-20 in case some fail)
- [ ] Cover all three lighting conditions across the set
- [ ] Keep each page to ~100-200 words so typing is manageable
- [ ] Type the exact text into `data/ground_truth/page_XX.txt` (UTF-8)
- [ ] Match the photo name: `page_01.txt` pairs with its photo; keep a `mapping.csv`
- [ ] A second teammate proofreads each ground-truth file (typos here corrupt the CER)

Decide and write down one rule for line breaks and extra whitespace. The
CER code will normalise whitespace, so type clean lines.

## C. SmartDoc (public set with known corners)

- [ ] Search "SmartDoc dataset ICDAR 2015" and download from the official page
- [ ] Check the license and note it for the report
- [ ] Extract a subset (50-100 frames) with their corner annotations
- [ ] Record the annotation format (corner order, coordinate units)

## D. Optional extras

- [ ] A few samples from Kaggle ("document images", "scanned documents")

## E. Repo and team tasks

- [ ] Create the GitHub repo and push this skeleton
- [ ] Everyone clones and installs `requirements.txt`
- [ ] Everyone confirms `tesseract --version` works
- [ ] Share photos via Drive; keep them out of git (`.gitignore` covers it)
- [ ] Split capture: each member photographs 15 images (so 60 total)

## Done when
- 60+ photos named and sorted in `data/raw/`
- 15+ pages with proofread ground truth
- SmartDoc subset downloaded with annotations parsed
- Repo pushed, everyone can run `tesseract --version`
