import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pulearn import ElkanotoPuClassifier


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

    return df_pred
