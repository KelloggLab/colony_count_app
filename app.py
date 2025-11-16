import io
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw

from streamlit_image_coordinates import streamlit_image_coordinates

import guiHelper as gh
import colony_count

#initialize colony_count model
model = 0
# --- Load image ---
IMAGE_PATH = "example2.png"  # change this to your image

img = Image.open(IMAGE_PATH)
# Resize to fit screen (e.g., max 1200px width)
MAX_WIDTH = 1200


draw = ImageDraw.Draw(gh.rescale_image(img,MAX_WIDTH))

# --- Initialize session state for points ---
if "points" not in st.session_state:
    st.session_state.points = []  # list of dicts: {"x": ..., "y": ...}

annotated = img.copy()
gh.draw_annotated_image(annotated,st)
# --- Get click coordinates on the *annotated* image ---
click = streamlit_image_coordinates(
    annotated,              # NOTE: we now pass the annotated image
    key="clickable-image",  # important so Streamlit tracks this widget
)

# If user clicked, store point
if click is not None:
    # click looks like {"x": int, "y": int, "time": float}
    st.session_state.points.append({"x": click["x"], "y": click["y"]})

st.write("Current points:", st.session_state.points)

# --- Clear points button ---
if st.button("Clear all points"):
    st.session_state.points = []
    st.experimental_rerun()


if st.button("Train Model"):
	df = pd.DataFrame(st.session_state.points)
	model = colony_count.train(df,img)

if st.button("Pick Using Model"):
	print("empty")
#	df = colony_count.pick(img,model)

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

