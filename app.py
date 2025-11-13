import io
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw

from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Point Annotation Demo", layout="centered")

st.title("Click-to-Annotate Image")

# --- Load image ---
IMAGE_PATH = "example.jpg"  # change this to your image
img = Image.open(IMAGE_PATH).convert("RGBA")

# --- Initialize session state for points ---
if "points" not in st.session_state:
    st.session_state.points = []  # list of dicts: {"x": ..., "y": ...}

st.write("Click on the image to add annotation points.")

# --- Get click coordinates ---
# You can pass a URL or a PIL image; here we use PIL
click = streamlit_image_coordinates(
    img,
    key="clickable-image",  # important so Streamlit tracks this widget
)

# If user clicked, store point
if click is not None:
    # click looks like {"x": int, "y": int, "time": float}
    st.session_state.points.append({"x": click["x"], "y": click["y"]})

st.write("Current points:", st.session_state.points)

# --- Draw markers on a copy of the image ---
annotated = img.copy()
draw = ImageDraw.Draw(annotated)

radius = 6
for idx, pt in enumerate(st.session_state.points):
    x, y = pt["x"], pt["y"]
    # draw a small circle
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline="red",
        width=2,
    )
    # optional: draw index label
    draw.text((x + radius + 2, y - radius - 2), str(idx + 1), fill="red")

st.image(annotated, caption="Annotated image", use_column_width=True)

# --- Clear points button ---
if st.button("Clear all points"):
    st.session_state.points = []
    st.experimental_rerun()

# --- Prepare downloads ---
if st.session_state.points:
    # 1) Download coordinates as CSV
    df = pd.DataFrame(st.session_state.points)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download coordinates (CSV)",
        data=csv_bytes,
        file_name="points.csv",
        mime="text/csv",
    )

    # 2) Download annotated image as PNG
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    buf.seek(0)
    st.download_button(
        label="⬇️ Download annotated image (PNG)",
        data=buf,
        file_name="annotated_image.png",
        mime="image/png",
    )

