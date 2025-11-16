import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pulearn import ElkanotoPuClassifier
import joblib
from PIL import Image, ImageDraw


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

def save_model(model, path):
    """
    Save a trained ColonyModel to disk.

    Parameters
    ----------
    model : ColonyModel
        The trained model returned by train().
    path : str or Path
        File path to save the model (e.g. 'colony_model.joblib').
    """
    obj = {
        "patch_radius": model.patch_radius,
        "classifier": model.classifier,
        "_is_trained": model._is_trained,
    }
    joblib.dump(obj, path)


def load_model(path):
    """
    Load a ColonyModel from disk.

    Parameters
    ----------
    path : str or Path
        File path where the model was saved.

    Returns
    -------
    ColonyModel
        A restored model instance with the same classifier and settings.
    """
    obj = joblib.load(path)
    model = ColonyModel(
        classifier=obj["classifier"],
        patch_radius=obj["patch_radius"],
    )
    # restore training flag if present
    if "_is_trained" in obj:
        model._is_trained = obj["_is_trained"]
    return model
    
import math
import numpy as np
from PIL import Image


def save_patch_gallery(patches, out_path, patch_radius=None, n_cols=10, pad=2):
    """
    Create a gallery image of patches and save as a single PNG.

    Parameters
    ----------
    patches : np.ndarray
        Array of shape (N, H, W) or (N, D).
        If (N, D), patches are assumed to be flattened square patches.
    out_path : str or Path
        Where to save the gallery image (e.g. 'patch_gallery.png').
    patch_radius : int or None
        If patches are flattened (N, D), and this is provided, patch size
        is taken as (2*patch_radius+1). If None, size is inferred as sqrt(D).
    n_cols : int
        Number of columns in the gallery grid.
    pad : int
        Padding (in pixels) between patches.
    """
    patches = np.asarray(patches)
    N = patches.shape[0]

    if N == 0:
        raise ValueError("No patches provided to save_patch_gallery.")

    # --- reshape to (N, H, W) if needed ---
    if patches.ndim == 2:
        # patches are flattened: (N, D)
        D = patches.shape[1]

        if patch_radius is not None:
            size = 2 * patch_radius + 1
            expected_D = size * size
            if D != expected_D:
                raise ValueError(
                    f"Patch dimension {D} does not match radius {patch_radius} "
                    f"(expected {expected_D})."
                )
        else:
            size = int(math.sqrt(D))
            if size * size != D:
                raise ValueError(
                    "Cannot infer square patch size from flattened dimension "
                    f"{D}. Provide patch_radius explicitly."
                )

        patches_reshaped = patches.reshape(N, size, size)
    elif patches.ndim == 3:
        # already (N, H, W)
        patches_reshaped = patches
        size = patches.shape[1]  # assume square
    else:
        raise ValueError(
            f"Unsupported patches shape {patches.shape}. "
            "Expected (N, D) or (N, H, W)."
        )

    H, W = patches_reshaped.shape[1], patches_reshaped.shape[2]

    # --- normalize to 0–255 and convert to uint8 for display ---
    pmin = patches_reshaped.min()
    pmax = patches_reshaped.max()
    if pmax > pmin:
        norm = (patches_reshaped - pmin) / (pmax - pmin)
    else:
        norm = np.zeros_like(patches_reshaped)
    norm_uint8 = (norm * 255).astype(np.uint8)

    # --- compute grid layout ---
    n_rows = math.ceil(N / n_cols)

    gallery_width = n_cols * W + (n_cols + 1) * pad
    gallery_height = n_rows * H + (n_rows + 1) * pad

    # single-channel (grayscale) gallery image
    gallery = Image.new("L", (gallery_width, gallery_height), color=0)

    # --- paste patches into the gallery ---
    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if idx >= N:
                break
            patch_img = Image.fromarray(norm_uint8[idx])
            x0 = pad + col * (W + pad)
            y0 = pad + row * (H + pad)
            gallery.paste(patch_img, (x0, y0))
            idx += 1

    # --- save to disk ---
    gallery.save(out_path, format="PNG")
    return gallery  # also return the PIL image in case you want to inspect it



def extract_patches_from_points(img, points, patch_radius=4, flatten=True):
    """
    Extract patches from an image at given (x, y) coordinates.

    Parameters
    ----------
    img : PIL.Image or ndarray
        Input image. Will be converted to grayscale internally.
    points : pandas.DataFrame or iterable of (x, y)
        If a DataFrame, must contain columns 'x' and 'y'.
        Otherwise, can be any iterable of (x, y) pairs.
    patch_radius : int
        Radius r of the patch. Each patch is (2r+1) x (2r+1).
    flatten : bool
        If True, returns patches flattened to shape (N, D),
        where D = (2r+1)^2. If False, returns (N, H, W).

    Returns
    -------
    np.ndarray
        Array of patches, either (N, D) if flatten=True or (N, H, W) otherwise.
        This can be passed directly to `save_patch_gallery`.
    """
    img_gray = _image_to_gray(img)

    # Normalize points input to a list of (x, y)
    if hasattr(points, "iterrows"):  # likely a DataFrame
        coords = [(int(row.x), int(row.y)) for _, row in points.iterrows()]
    else:
        coords = [(int(x), int(y)) for (x, y) in points]

    size = 2 * patch_radius + 1

    patches = []
    for (x, y) in coords:
        patch_flat = _extract_patch(img_gray, x, y, patch_radius=patch_radius)
        if flatten:
            patches.append(patch_flat)
        else:
            patches.append(patch_flat.reshape(size, size))

    patches = np.array(patches)
    return patches


import numpy as np
import cv2
from PIL import Image


def detect_plate_circle(
    img,
    diameter_ratio=0.9,
    diameter_tolerance=0.15,
    dp=1.2,
    edge_blur_ksize=5,
):
    """
    Detect a circular plate whose diameter is expected to be a fixed fraction
    of the image size, using a Hough circle transform.

    Parameters
    ----------
    img : PIL.Image or np.ndarray
        Input image (RGB, BGR, or grayscale).
    diameter_ratio : float
        Expected diameter / min(image_width, image_height).
        For example, 0.9 means the plate occupies ~90% of the smaller dimension.
    diameter_tolerance : float
        Fractional tolerance around the expected diameter.
        The search radius range is:
            R_expected * (1 - diameter_tolerance) -> minRadius
            R_expected * (1 + diameter_tolerance) -> maxRadius
    dp : float
        Inverse ratio of the accumulator resolution to the image resolution
        (HoughCircles parameter). 1.2 is a good default.
    edge_blur_ksize : int
        Kernel size for median blur applied before edge detection.

    Returns
    -------
    (cx, cy, R) or None
        Center (cx, cy) and radius R in pixel coordinates, or None if no circle found.
    """
    # Convert image to grayscale np.ndarray
    if isinstance(img, Image.Image):
        arr = np.array(img)
    else:
        arr = np.asarray(img)

    if arr.ndim == 3:
        # assume RGB
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr.astype(np.uint8)

    # Basic smoothing to help edge detection
    gray_blur = cv2.medianBlur(gray, edge_blur_ksize)

    # Determine size-based parameters
    h, w = gray_blur.shape[:2]
    base_dim = min(h, w)

    # Expected diameter and radius from ratio
    expected_diameter = diameter_ratio * base_dim
    expected_radius = expected_diameter / 2.0

    # Radius search band based on tolerance
    min_radius = int(expected_radius * (1.0 - diameter_tolerance))
    max_radius = int(expected_radius * (1.0 + diameter_tolerance))

    # Ensure they are valid
    min_radius = max(1, min_radius)
    if max_radius <= min_radius:
        max_radius = min_radius + 5

    # minDist: minimum distance between circle centers
    # For a single plate, we can set this fairly large (e.g. half of base_dim)
    min_dist = int(base_dim * 0.5)

    # Hough parameters:
    # param1 is the upper threshold for the internal Canny edge detector
    param1 = 100

    # param2 is the accumulator threshold for center detection.
    # Grow it mildly with radius so larger circles require more supporting edges.
    param2 = max(20, int(expected_radius * 0.05))

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return None

    circles = np.squeeze(circles, axis=0)  # shape (N, 3): (x, y, r)

    # Choose the largest circle (likely the plate)
    idx = np.argmax(circles[:, 2])
    x, y, r = circles[idx]

    return float(x), float(y), float(r)


def remove_edge_plate_points(df_points, cx, cy, R, edge_margin=10):
    """
    Keep only points that are at least `edge_margin` pixels inside
    the plate (a circle of radius R centered at (cx, cy)).
    """
    df = df_points.copy()
    dx = df["x"] - cx
    dy = df["y"] - cy
    r = np.sqrt(dx**2 + dy**2)
    max_r = R - edge_margin
    return df[r <= max_r].reset_index(drop=True)


# -------------------------
# Internal helper functions
# -------------------------

def _extract_patch(image, x, y, patch_radius=4):
    """
    Extracts a (2r+1) x (2r+1) square patch centered at (x,y)
    from a 2D numpy array. Pads if near borders.
    """
    H, W = image.shape
    size = 2 * patch_radius + 1

    x0 = max(0, x - patch_radius)
    x1 = min(W, x + patch_radius + 1)
    y0 = max(0, y - patch_radius)
    y1 = min(H, y + patch_radius + 1)

    patch = image[y0:y1, x0:x1]

    # pad to fixed shape
    padded = np.zeros((size, size), dtype=image.dtype)
    padded[:patch.shape[0], :patch.shape[1]] = patch

    return padded.flatten()


def _image_to_gray(img):
    """
    Convert PIL image or ndarray to grayscale numpy array (H,W).
    """
    arr = np.array(img)
    if arr.ndim == 3:  # RGB
        arr = arr.mean(axis=2)
    return arr.astype("float32")


# -------------------------
# Model wrapper class
# -------------------------

class ColonyModel:
    """
    A simple wrapper around the PU-learning classifier.
    """

    def __init__(self, classifier, patch_radius):
        self.classifier = classifier
        self.patch_radius = patch_radius
        self._is_trained = True

    def is_valid(self):
        """
        Returns True if the model is trained and ready.
        """
        return self._is_trained is True


# -------------------------
# Public API
# -------------------------

def train(df_points, img, patch_radius=4, num_unlabeled=5000, random_state=42):
    """
    Train a positive-unlabeled (PU) model based on user-clicked points.

    Parameters:
        df_points: DataFrame with columns ["x","y"]
        img: PIL Image or ndarray (will be converted to grayscale)
        patch_radius: size of local neighborhood
        num_unlabeled: number of unlabeled samples to draw
        random_state: reproducibility

    Returns:
        ColonyModel
    """
    img_gray = _image_to_gray(img)
    H, W = img_gray.shape
    rng = np.random.default_rng(random_state)

    # Positives
    pos_coords = [(int(row.x), int(row.y)) for _, row in df_points.iterrows()]
    X_pos = np.array([
        _extract_patch(img_gray, x, y, patch_radius)
        for (x, y) in pos_coords
    ])

    # Unlabeled sampling
    all_coords = [(x, y) for y in range(H) for x in range(W)]
    pos_set = set(pos_coords)
    unlabeled_candidates = [(x, y) for (x, y) in all_coords if (x, y) not in pos_set]

    n_unl = min(num_unlabeled, len(unlabeled_candidates))
    idxs = rng.choice(len(unlabeled_candidates), size=n_unl, replace=False)
    unl_coords = [unlabeled_candidates[i] for i in idxs]

    X_unl = np.array([
        _extract_patch(img_gray, x, y, patch_radius)
        for (x, y) in unl_coords
    ])

    # Training data
    X_train = np.vstack([X_pos, X_unl])
    y_train = np.hstack([np.ones(X_pos.shape[0]), np.zeros(X_unl.shape[0])])

    # Base classifier
    base_clf = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=random_state,
    )

    # PU wrapper
    pu_clf = ElkanotoPuClassifier(
        estimator=base_clf,
        hold_out_ratio=0.2,
        random_state=random_state
    )

    pu_clf.fit(X_train, y_train)

    # Wrap in ColonyModel
    model = ColonyModel(pu_clf, patch_radius)
    return model


def preprocess_for_circles(img, sigma=2.0, ksize=0):
    """
    Convert to grayscale and apply Gaussian low-pass filter.
    img: PIL or ndarray (RGB or grayscale)
    sigma: Gaussian sigma in pixels
    ksize: kernel size; 0 lets OpenCV pick based on sigma
    """
    # to ndarray
    if isinstance(img, Image.Image):
        arr = np.array(img)
    else:
        arr = np.asarray(img)

    # RGB → grayscale
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr.astype(np.uint8)

    # Gaussian low-pass filter
    # If ksize == 0, OpenCV infers a suitable kernel size from sigma
    gray_blur = cv2.GaussianBlur(gray, (ksize or 0, ksize or 0), sigmaX=sigma)

    return gray_blur

def pick(img, model, threshold=0.9):
    """
    Apply the trained model to an image and return predicted feature coordinates.

    Parameters:
        img: PIL Image or ndarray
        model: ColonyModel returned by train()
        threshold: probability threshold for calling positives

    Returns:
        DataFrame with predicted points (x,y)
    """
    if not model.is_valid():
        raise ValueError("Model is not valid. Please train first.")

    img_gray = _image_to_gray(img)
    H, W = img_gray.shape
    patch_radius = model.patch_radius

    # Extract patches for entire image
    coords = [(x, y) for y in range(H) for x in range(W)]
    X_all = np.array([
        _extract_patch(img_gray, x, y, patch_radius)
        for (x, y) in coords
    ])

    # Predict probabilities
    probs = model.classifier.predict_proba(X_all)[:, 1]
    prob_map = probs.reshape(H, W)

    # Threshold
    ys, xs = np.where(prob_map >= threshold)
    df_pred = pd.DataFrame({"x": xs.astype(int), "y": ys.astype(int)})
    
    #NOTE FIX THIS LATER
    lpimg = preprocess_for_circles(img,4,0)
    testout = Image.fromarray(lpimg)
    testout.save('lowpass_img.png',format="PNG")
    cx, cy, R = detect_plate_circle(lpimg)
    print("Detected plate center & radius:", cx, cy, R)

	# 2) Remove edge artifacts from your annotated or predicted points
    df_clean = remove_edge_plate_points(df_pred, cx, cy, R, edge_margin=20)
    #check
    annotated_img = draw_points_on_image(img, df_pred, radius=3, color="green")
    annotated_img = draw_points_on_image(annotated_img, df_clean, radius=3, color="red")
    annotated_img.save('remove_edge_check.png', format="PNG")
    #END
    

    return df_clean
