import io
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw

from streamlit_image_coordinates import streamlit_image_coordinates
import colony_count

# make page wide so 1200 px images fit without cropping
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# initialize colony_count model
model = 0

# --- Load image ---
IMAGE_PATH = st.file_uploader("plate image:", type=["jpg", "png"])

if IMAGE_PATH is not None:
    img = Image.open(IMAGE_PATH)

    # resize to fit screen, keep aspect ratio
    MAX_WIDTH = 1200
    if img.width > MAX_WIDTH:
        scale = MAX_WIDTH / img.width
        new_size = (MAX_WIDTH, int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)

    st.caption(f"Image size used by app: {img.width} x {img.height}")

    # initialize session state for points
    if "points" not in st.session_state:
        st.session_state.points = []  # list of dicts: {"x": ..., "y": ...}

    # draw existing manual points on a copy of the image
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    radius_vis = 6
    for pt in st.session_state.points:
        x, y = pt["x"], pt["y"]
        draw.ellipse(
            (x - radius_vis, y - radius_vis, x + radius_vis, y + radius_vis),
            outline="red",
            width=2,
        )

    # single clickable image
    click = streamlit_image_coordinates(
        annotated,
        key="clickable-image",
        width=annotated.width,
        height=annotated.height,
    )

    # if user clicked, store point
    if click is not None:
        st.session_state.points.append({"x": click["x"], "y": click["y"]})

    st.write("Current points:", st.session_state.points)

    # clear points button
    if st.button("Clear all points"):
        st.session_state.points = []
        st.experimental_rerun()

    radius = st.number_input(
        "size of patch for feature extraction (pixels): ",
        min_value=1,
        max_value=20,
        value=4,
    )

    if st.button("Train Model"):
        df = pd.DataFrame(st.session_state.points)
        model = colony_count.train(df, img)
        colony_count.save_model(model, "trained_model.joblib")

    threshold = st.number_input(
        "threshold for predictions: ",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
    )

    sigma = st.number_input(
        "sigma for gaussian filter (for plate edge detection): ",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
    )

    edge_margin = st.number_input(
        "edge margin for removal of points, in pixels: ",
        min_value=1,
        max_value=50,
        value=30,
    )

    diameter_ratio = st.number_input(
        "ratio of diameter of plate / image width: ",
        min_value=0.01,
        max_value=1.0,
        value=0.9,
    )

    # start with manual-annotated image as default for download
    annotated_img = annotated.copy()

    if st.button("Pick Using Model"):
        model = colony_count.load_model("trained_model.joblib")
        df_pred = colony_count.pick(
            img,
            model,
            threshold,
            sigma,
            edge_margin,
            diameter_ratio,
        )
        annotated_img = colony_count.draw_points_on_image(
            img, df_pred, radius=3, color="red"
        )

        st.image(
            annotated_img,
            caption=f"Model-picked colonies (size: {annotated_img.width} x {annotated_img.height})",
        )
        st.title("number of colonies detected: " + str(len(df_pred)))

    # downloads
    if st.session_state.points:
        # coordinates csv
        df_points = pd.DataFrame(st.session_state.points)
        csv_bytes = df_points.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download coordinates (CSV)",
            data=csv_bytes,
            file_name="points.csv",
            mime="text/csv",
        )

        # annotated image png (manual or model, depending on last state)
        buf = io.BytesIO()
        annotated_img.save(buf, format="PNG")
        buf.seek(0)
        st.download_button(
            label="⬇️ Download annotated image (PNG)",
            data=buf,
            file_name="annotated_image.png",
            mime="image/png",
        )

else:
    st.info("Please upload an image to begin.")

