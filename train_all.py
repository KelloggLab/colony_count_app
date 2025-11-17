from pathlib import Path
import pandas as pd
from PIL import Image
import colony_count

image_paths = ["test1.jng", "test2.jpg"]
csv_paths   = ["points1.csv", "points2.csv"]

images = [Image.open(p) for p in image_paths]
df_list = [pd.read_csv(p) for p in csv_paths]

model = colony_count.train_all(images, df_list, patch_radius=4, num_unlabeled_per_image=5000)
colony_count.save_model(model, "colony_model_multi.joblib")

