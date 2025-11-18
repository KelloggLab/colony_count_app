import io
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw

from streamlit_image_coordinates import streamlit_image_coordinates

# ---------- basic page setup (optional but nice) ----------
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

st.title("CFU colony annotator")
st.write("Click on the plate image to mark colonies. You can download the coordinates and annotated image.")

# --- Load image ---
IMAGE_PATH = "plate_original.jpg"  # <--- use your cropped/square plate image here

img = Image.open(IMAGE_PATH)

# Resize to fit screen (e.g., max 1200px width)
MAX_WIDTH = 1200
if img.width > MAX_WIDTH:
    scale = MAX_WIDTH / img.width
    new_size = (MAX_WIDTH, int(img.height * scale))
    img = img.resize(new_size, Image.LANCZOS)

st.caption(f"Image size used by app: {img.width} x {img.height}")

# --- Initialize session state for points + IDs ---
if "points" not in st.session_state:
    # list of dicts: {"id": int, "x": int, "y": int}
    st.session_state.points = []

if "next_id" not in st.session_state:
    st.session_state.next_id = 1  # ensures labels go 1,2,3,...

# --- Draw markers on a copy of the image using current points ---
annotated = img.copy()
draw = ImageDraw.Draw(annotated)

radius = 6
for pt in st.session_state.points:
    x, y = pt["x"], pt["y"]
    label = str(pt["id"])
    # draw small circle
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline="red",
        width=2,
    )
    # index label
    draw.text((x + radius + 2, y - radius - 2), label, fill="red")

# --- Get click coordinates on the annotated image ---
click = streamlit_image_coordinates(
    annotated,
    key="clickable-image",
    width=annotated.width,   # make widget match real size
    height=annotated.height,
)

# If user clicked, store point with sequential ID
if click is not None:
    st.session_state.points.append(
        {"id": st.session_state.next_id, "x": click["x"], "y": click["y"]}
    )
    st.session_state.next_id += 1

# Show current points in a nice table
if st.session_state.points:
    df = pd.DataFrame(st.session_state.points)[["id", "x", "y"]]
    df = df.rename(columns={"id": "colony_id"})
    st.write(f"Current points (total: {len(df)}):")
    st.dataframe(df)
else:
    st.write("Current points: []")

# --- Clear points button ---
if st.button("Clear all points"):
    st.session_state.points = []
    st.session_state.next_id = 1
    st.experimental_rerun()

# --- Prepare downloads ---
if st.session_state.points:
    df = pd.DataFrame(st.session_state.points)[["id", "x", "y"]]
    df = df.rename(columns={"id": "colony_id"})

    # 1) Download coordinates as CSV
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

