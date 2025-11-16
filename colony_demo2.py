import argparse
import pandas as pd
from PIL import Image, ImageDraw
import colony_count


def draw_points_on_image(img, df_points, radius=4, color="red"):
    """
    Draw circular markers for each (x, y) in df_points on a copy of img.

    Parameters
    ----------
    img : PIL.Image
        Base image (will not be modified in place).
    df_points : DataFrame
        Must contain columns 'x' and 'y'.
    radius : int
        Radius of the drawn circles.
    color : str or tuple
        Color for the circles.

    Returns
    -------
    PIL.Image
        Annotated image.
    """
    annotated = img.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    for _, row in df_points.iterrows():
        x, y = int(row["x"]), int(row["y"])
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=color,
            width=2,
        )

    return annotated


def main():
    parser = argparse.ArgumentParser(
        description="Train PU model from CSV annotations and predict new features."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image (e.g. colonies.jpg)",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to CSV with annotated features (columns: x,y)",
    )
    parser.add_argument(
        "--output",
        default="predicted_annotations.png",
        help="Path to output annotated PNG image",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Probability threshold for calling positives (default: 0.9)",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=4,
        help="radius for patch extraction"),

    args = parser.parse_args()

    # --- Load image ---
    img = Image.open(args.image)

    # --- Load annotated features ---
    df_points = pd.read_csv(args.csv)

    if not {"x", "y"}.issubset(df_points.columns):
        raise ValueError("CSV must contain at least 'x' and 'y' columns.")

    print(f"Loaded image: {args.image}")
    print(f"Loaded {len(df_points)} annotated points from: {args.csv}")

    # --- Train model ---
    print("Training model...")
    model = colony_count.train(df_points, img,args.radius)
    print(f"Model trained. is_valid(): {model.is_valid()}")

    # --- Predict new features ---
    print("Predicting new features on the image...")
    df_pred = colony_count.pick(img, model, threshold=args.threshold)
    print(f"Predicted {len(df_pred)} positive pixels at threshold {args.threshold}.")

    # --- Draw predictions onto the original image ---
    annotated_img = draw_points_on_image(img, df_pred, radius=3, color="red")

    # --- Save annotated image as PNG ---
    annotated_img.save(args.output, format="PNG")
    print(f"Annotated image with predictions saved to: {args.output}")


if __name__ == "__main__":
    main()

