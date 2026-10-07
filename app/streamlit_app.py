"""Phase 6: Streamlit page.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `src` imports work

import cv2
import numpy as np
import streamlit as st

from src.enhance import METHODS
from src.pipeline import DEFAULT_METHOD, scan

st.set_page_config(page_title="Document Scanner", layout="wide")
st.title("Smart Document Scanner and Enhancer")
st.caption("Upload a phone photo of a page. It is flattened, cleaned and can be read with OCR.")


@st.cache_resource
def get_ocr():
    """OCR is optional: the app still works if Tesseract is not installed."""
    try:
        from src.ocr_eval import run_ocr
        return run_ocr
    except SystemExit:
        return None


def to_png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


with st.sidebar:
    st.header("Settings")
    method = st.selectbox("Enhancement", list(METHODS), index=list(METHODS).index(DEFAULT_METHOD))
    upscale = st.slider("Upscale after flattening", 1.0, 3.0, 1.0, 0.5,
                        help="Enlarging can help OCR on small text.")
    strategy = st.selectbox("Page detection", ["largest", "robust"],
                            help="largest = textbook method; robust = scores several candidates")
    show_all = st.checkbox("Compare all methods", value=False)
    do_ocr = st.checkbox("Read text (OCR)", value=False)
    st.markdown("---")
    st.caption("Classroom project. Photos are processed in memory and not stored.")

upload = st.file_uploader("Photo of a page", type=["jpg", "jpeg", "png", "bmp", "webp"])

if upload is None:
    st.info("Choose a photo. Keep all four page corners in the frame, with some desk visible around the page.")
    st.stop()

data = np.frombuffer(upload.read(), dtype=np.uint8)
photo = cv2.imdecode(data, cv2.IMREAD_COLOR)
if photo is None:
    st.error("Could not read that image.")
    st.stop()

with st.spinner("Scanning..."):
    result = scan(photo, upscale=upscale, strategy=strategy)

if result.found:
    st.success("Page found.")
else:
    st.warning(f"The page outline was not found cleanly (method: {result.detection.method}). "
               "The result may be poor. Try a photo with the whole page and some background visible.")

col1, col2, col3 = st.columns(3)
col1.subheader("1. Detected page")
col1.image(cv2.cvtColor(result.detection.debug["overlay"], cv2.COLOR_BGR2RGB), use_container_width=True)
col2.subheader("2. Flattened")
col2.image(cv2.cvtColor(result.warped, cv2.COLOR_BGR2RGB), use_container_width=True)
out = result.output(method)
col3.subheader(f"3. Clean scan ({method})")
col3.image(out, use_container_width=True, clamp=True)

st.download_button("Download scan (PNG)", to_png_bytes(out), file_name=f"{Path(upload.name).stem}_scan.png",
                   mime="image/png")

if show_all:
    st.subheader("All methods")
    cols = st.columns(3)
    for i, (name, img) in enumerate(result.enhanced.items()):
        cols[i % 3].image(img, caption=name, use_container_width=True, clamp=True)

if do_ocr:
    run_ocr = get_ocr()
    if run_ocr is None:
        st.error("Tesseract is not installed or not found, so OCR is unavailable.")
    else:
        with st.spinner("Reading text..."):
            text = run_ocr(out)
        st.subheader("Extracted text")
        st.text_area("OCR result", text, height=250)