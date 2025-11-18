from pathlib import Path
import pandas as pd
from PIL import Image
import colony_count
import argparse


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train colony counter")

    parser.add_argument(
        "--images",
        required=True,
        type=str,
        help="Path to a text file containing a list of image paths"
    )

    parser.add_argument(
        "--picked_points",
        required=True,
        type=str,
        help="Path to a text file containing a list of CSV files with picked points"
    )

    return parser.parse_args()


def load_list(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


args = parse_args()

image_paths = load_list(args.images)
csv_paths   = load_list(args.picked_points)

images = [Image.open(p) for p in image_paths]
df_list = [pd.read_csv(p) for p in csv_paths]

model = colony_count.train_all(images, df_list, patch_radius=4, num_unlabeled_per_image=5000)
colony_count.save_model(model, "colony_model_multi.joblib")

