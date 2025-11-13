
import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw

from streamlit_image_coordinates import streamlit_image_coordinates


def draw_annotated_image(annotated,st):
	# --- Draw markers on a copy of the image using current points ---
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

def rescale_image(img,MAX_WIDTH):
	if img.width > MAX_WIDTH:
		scale = MAX_WIDTH / img.width
		new_size = (MAX_WIDTH, int(img.height * scale))
		img = img.resize(new_size, Image.LANCZOS)
	return img