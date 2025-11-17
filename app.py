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
#IMAGE_PATH = "training_set/test2.jpg"  # change this to your image
IMAGE_PATH = st.file_uploader("plate image:", type=["jpg","png"])

if IMAGE_PATH is not None:
	img = Image.open(IMAGE_PATH)
	# Resize to fit screen (e.g., max 1200px width)
	MAX_WIDTH = 1200

	draw = ImageDraw.Draw(gh.rescale_image(img,MAX_WIDTH))

	# --- Initialize session state for points ---
	if "points" not in st.session_state:
		st.session_state.points = []  # list of dicts: {"x": ..., "y": ...}

	annotated = img.copy()
	annotated_img = img.copy()
	gh.draw_annotated_image(annotated,st)
	# --- Get click coordinates on the *annotated* image ---
	click = streamlit_image_coordinates(
    	annotated,              # NOTE: we now pass the annotated image
    	key="clickable-image",  # important so Streamlit tracks this widget
	)

	# If user clicked, store point
	if click is not None:
		st.session_state.points.append({"x": click["x"], "y": click["y"]})

	st.write("Current points:", st.session_state.points)

	# --- Clear points button ---
	if st.button("Clear all points"):
		st.session_state.points = []
		st.experimental_rerun()

	radius = st.number_input(
		"size of patch for feature extraction (pixels): ",
		min_value=1,
		max_value=20,
		value=4)
	
	if st.button("Train Model"):
		df = pd.DataFrame(st.session_state.points)
		model = colony_count.train(df,img)
		colony_count.save_model(model,'trained_model.joblib')
	
	threshold = st.number_input(
		"threshold for predictions: ",
		min_value=0.0,
		max_value=1.0,
		value=0.2)

	sigma = st.number_input(
		"sigma for gaussian filter (for plate edge detection): ",
		min_value=0.0,
		max_value=10.0,
		value=2.0)
		
	edge_margin = st.number_input(
		"edge margin for removal of points, in pixels: ",
		min_value=1,
		max_value=50,
		value=30)
		
	diameter_ratio = st.number_input(
		"ratio of diameter of plate / image width: ",
		min_value=0.01,
		max_value=1.0,
		value=0.9)

	if st.button("Pick Using Model"):
		model = colony_count.load_model('trained_model.joblib')
		df = colony_count.pick(img,
								model,
								threshold,
								sigma,
								edge_margin,
								diameter_ratio)
		annotated_img = colony_count.draw_points_on_image(img, df, radius=3, color="red")
		gh.draw_annotated_image(annotated_img,st)
		click = streamlit_image_coordinates(
    		annotated_img,              # NOTE: we now pass the annotated image
    		key="clickable-image2",  # important so Streamlit tracks this widget
		)
		st.title('number of colonies detected: '+str(len(df)))


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