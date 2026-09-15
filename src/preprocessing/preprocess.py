"""
MRI Image Preprocessing
-----------------------
Handles:
  - Loading images from disk
  - Skull-stripping approximation (Otsu thresholding)
  - Normalization
  - Resizing to model input size
  - Data augmentation pipeline (training only)
"""

import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config


# ─── Core Utilities ───────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load an image as BGR uint8 numpy array."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at: {path}")
    return img


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def skull_strip(img: np.ndarray) -> np.ndarray:
    """
    Approximate skull stripping using Otsu thresholding + largest contour.
    Removes non-brain background. Works well on T1/T1C MRI images.

    Args:
        img: RGB uint8 image (H, W, 3)
    Returns:
        Skull-stripped image with background set to 0
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Otsu threshold to separate brain from skull/background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Keep only the largest connected component (the brain)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img  # fallback — return original if no contours found

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

    # Apply mask to all 3 channels
    stripped = img.copy()
    stripped[mask == 0] = 0
    return stripped


def normalize(img: np.ndarray) -> np.ndarray:
    """Scale pixel values to [0, 1] float32."""
    return img.astype(np.float32) / 255.0


def resize_classification(img: np.ndarray) -> np.ndarray:
    """Resize for EfficientNetB0 input (224×224)."""
    return cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT), interpolation=cv2.INTER_AREA)


def resize_segmentation(img: np.ndarray) -> np.ndarray:
    """Resize for U-Net input (256×256)."""
    return cv2.resize(img, (config.SEG_IMG_WIDTH, config.SEG_IMG_HEIGHT), interpolation=cv2.INTER_AREA)


# ─── Full Preprocessing Pipelines ─────────────────────────────────────────────

def preprocess_for_classification(image_path: str, apply_skull_strip: bool = True) -> np.ndarray:
    """
    Full preprocessing pipeline for classification.

    Args:
        image_path: Path to MRI image file
        apply_skull_strip: Whether to apply skull stripping

    Returns:
        Preprocessed image as float32 array of shape (224, 224, 3), values in [0,1]
    """
    img = load_image(image_path)
    img = bgr_to_rgb(img)

    if apply_skull_strip:
        img = skull_strip(img)

    img = resize_classification(img)
    img = normalize(img)
    return img


def preprocess_for_segmentation(image_path: str, mask_path: str = None):
    """
    Full preprocessing pipeline for segmentation training.

    Args:
        image_path: Path to MRI image
        mask_path: Path to ground truth mask (optional, for training)

    Returns:
        Tuple (image, mask) or just image if no mask provided.
        image: float32 (256, 256, 3) in [0,1]
        mask: float32 (256, 256, 1) binary in {0, 1}
    """
    img = load_image(image_path)
    img = bgr_to_rgb(img)
    img = resize_segmentation(img)
    img = normalize(img)

    if mask_path is not None:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load mask at: {mask_path}")
        mask = cv2.resize(mask, (config.SEG_IMG_WIDTH, config.SEG_IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)   # binarize
        mask = np.expand_dims(mask, axis=-1)      # (256, 256, 1)
        return img, mask

    return img


# ─── Augmentation ─────────────────────────────────────────────────────────────

def get_classification_augmentor() -> A.Compose:
    """
    Augmentation pipeline for classification training.
    Medical image augmentation — no color jitter (MRI is grayscale-based).
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
    ])


def get_segmentation_augmentor() -> A.Compose:
    """
    Augmentation for segmentation — must apply same transform to image AND mask.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=15, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    ], additional_targets={"mask": "mask"})


def apply_classification_augmentation(img: np.ndarray, augmentor: A.Compose) -> np.ndarray:
    """Apply augmentation to a single classification image."""
    # albumentations expects uint8
    img_uint8 = (img * 255).astype(np.uint8)
    augmented = augmentor(image=img_uint8)
    return augmented["image"].astype(np.float32) / 255.0


def apply_segmentation_augmentation(img: np.ndarray, mask: np.ndarray, augmentor: A.Compose):
    """Apply augmentation consistently to image and mask."""
    img_uint8 = (img * 255).astype(np.uint8)
    mask_uint8 = (mask[:, :, 0] * 255).astype(np.uint8)

    augmented = augmentor(image=img_uint8, mask=mask_uint8)
    aug_img = augmented["image"].astype(np.float32) / 255.0
    aug_mask = (augmented["mask"] > 127).astype(np.float32)
    aug_mask = np.expand_dims(aug_mask, axis=-1)
    return aug_img, aug_mask
