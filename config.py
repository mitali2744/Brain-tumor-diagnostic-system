"""
Central configuration for the Brain Tumor Detection System.
All paths, hyperparameters, and constants live here.
"""

import os

# ─── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ─── Dataset Paths ────────────────────────────────────────────────────────────
CLASSIFICATION_DATA_DIR = os.path.join(DATA_DIR, "Brain Tumor MRI Dataset")
SEGMENTATION_DATA_DIR = os.path.join(DATA_DIR, "LGG MRI Segmentation", "kaggle_3m")

# ─── Model Save Paths ─────────────────────────────────────────────────────────
CLASSIFIER_MODEL_PATH = os.path.join(MODELS_DIR, "efficientnetb0_classifier.h5")
SEGMENTATION_MODEL_PATH = os.path.join(MODELS_DIR, "unet_segmentation.h5")

# ─── Image Config ─────────────────────────────────────────────────────────────
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3          # RGB for classifier; grayscale converted to 3ch
SEG_IMG_HEIGHT = 256      # U-Net works well at 256×256
SEG_IMG_WIDTH = 256
SEG_IMG_CHANNELS = 3

# ─── Classification Config ────────────────────────────────────────────────────
NUM_CLASSES = 4
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_LABELS = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}

# ─── Training Hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS_CLASSIFIER = 30
EPOCHS_SEGMENTATION = 50
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2

# ─── Grad-CAM Config ──────────────────────────────────────────────────────────
GRADCAM_LAYER = "top_conv"          # Last conv layer in EfficientNetB0
GRADCAM_ALPHA = 0.4                 # Heatmap blend strength

# ─── Flask / App Config ───────────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(BASE_DIR, "app", "static", "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "app", "static", "results")
DATABASE_PATH = os.path.join(BASE_DIR, "app", "database", "brain_tumor.db")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024   # 16 MB max upload

# ─── Segmentation Threshold ───────────────────────────────────────────────────
SEG_THRESHOLD = 0.5     # Pixel probability above this → tumor
