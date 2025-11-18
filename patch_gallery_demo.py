import argparse
import pandas as pd
from PIL import Image

import colony_count  # this must define extract_patches_from_points & save_patch_gallery


def main():
    parser = argparse.ArgumentParser(
        description="Extract patches from image using coordinates in CSV and save as a gallery image."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image (e.g. colonies.png)",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to CSV file with coordinates (columns: x,y)",
    )
    parser.add_argument(
        "--output",
        default="patch_gallery.png",
        help="Path to output gallery PNG (default: patch_gallery.png)",
    )
    parser.add_argument(
        "--patch_radius",
        type=int,
        default=4,
        help="Patch radius r (patch size = (2r+1)x(2r+1); default: 4)",
    )
    parser.add_argument(
        "--n_cols",
        type=int,
        default=10,
        help="Number of columns in the gallery grid (default: 10)",
    )

    args = parser.parse_args()

    # --- Load image ---
    img = Image.open(args.image)
    print(f"Loaded image: {args.image}")

    # --- Load coordinates ---
    df_points = pd.read_csv(args.csv)
    if not {"x", "y"}.issubset(df_points.columns):
        raise ValueError("CSV must contain at least 'x' and 'y' columns.")

    print(f"Loaded {len(df_points)} points from: {args.csv}")

    # --- Extract patches from those points ---
    patches = colony_count.extract_patches_from_points(
        img,
        df_points,
        patch_radius=args.patch_radius,
        flatten=False,  # 2D patches (N, H, W) → fine for save_patch_gallery
    )

    print(f"Extracted {patches.shape[0]} patches of size {patches.shape[1]}x{patches.shape[2]}.")

    # --- Save gallery image ---
    colony_count.save_patch_gallery(
        patches,
        out_path=args.output,
        n_cols=args.n_cols,
        pad=2,
    )

    print(f"Saved patch gallery image to: {args.output}")


if __name__ == "__main__":
    main()
