"""
Inference Pipeline
-------------------
Central module called by Flask routes to run the full pipeline on an uploaded MRI:
  1. Preprocess image
  2. Classify (EfficientNetB0)
  3. Segment  (U-Net)
  4. Grad-CAM
  5. Save all output images
  6. Return structured result dict

Models are loaded once at startup (module-level singletons) to avoid
re-loading on every request.
"""

import os
import sys
import uuid
import json
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.preprocessing.preprocess import preprocess_for_classification, preprocess_for_segmentation
from src.classification.efficientnet_model import (
    load_classifier,
    predict_tumor_type,
    prepare_classifier_input,
)
from src.segmentation.unet import load_unet
from src.explainability.gradcam import generate_gradcam_visualization, find_last_conv_layer
from src.utils.visualization import save_mask_as_image, create_result_composite

os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(config.RESULTS_FOLDER, exist_ok=True)

# â”€â”€â”€ Singleton model loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_classifier = None
_segmentor = None
_gradcam_layer = None


def get_classifier():
    global _classifier
    if _classifier is None:
        if not os.path.exists(config.CLASSIFIER_MODEL_PATH):
            raise FileNotFoundError(
                f"Classifier model not found at {config.CLASSIFIER_MODEL_PATH}. "
                "Run train_classification.py first."
            )
        print("[Pipeline] Loading classifier model...")
        _classifier = load_classifier(config.CLASSIFIER_MODEL_PATH)
    return _classifier


def get_segmentor():
    global _segmentor
    if _segmentor is None:
        if not os.path.exists(config.SEGMENTATION_MODEL_PATH):
            raise FileNotFoundError(
                f"Segmentation model not found at {config.SEGMENTATION_MODEL_PATH}. "
                "Run train_segmentation.py first."
            )
        print("[Pipeline] Loading segmentation model...")
        _segmentor = load_unet(config.SEGMENTATION_MODEL_PATH)
    return _segmentor


def get_gradcam_layer():
    global _gradcam_layer
    if _gradcam_layer is None:
        try:
            # Try the default config layer name first
            classifier = get_classifier()
            classifier.get_layer(config.GRADCAM_LAYER)
            _gradcam_layer = config.GRADCAM_LAYER
        except ValueError:
            # Auto-detect last conv layer
            _gradcam_layer = find_last_conv_layer(get_classifier())
            print(f"[Pipeline] Auto-detected Grad-CAM layer: {_gradcam_layer}")
    return _gradcam_layer


# â”€â”€â”€ Core pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_pipeline(image_path: str, original_filename: str) -> dict:
    """
    Run the full inference pipeline on a single MRI image.

    Args:
        image_path: Absolute path to the uploaded MRI file
        original_filename: Original filename for display

    Returns:
        result dict with keys:
          - predicted_label, predicted_class, confidence, all_probabilities
          - upload_path, segmentation_path, gradcam_path, composite_path
          - (all *_path values are relative to app/static/ for url_for usage)
    """
    scan_id = str(uuid.uuid4())[:8]

    # â”€â”€ Step 1: Preprocess for classification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    clf_img = preprocess_for_classification(image_path, apply_skull_strip=False)

    # â”€â”€ Step 2: Preprocess for segmentation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    seg_img = preprocess_for_segmentation(image_path)   # (256, 256, 3) float32

    # â”€â”€ Step 3: Classification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    classifier = get_classifier()
    classifier_img = prepare_classifier_input(classifier, clf_img)
    prediction = predict_tumor_type(classifier, classifier_img)

    # â”€â”€ Step 4: Segmentation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    segmentor = get_segmentor()
    seg_input = np.expand_dims(seg_img, axis=0)                       # (1, 256, 256, 3)
    seg_prob   = segmentor.predict(seg_input, verbose=0)[0]            # (256, 256, 1)
    seg_binary = (seg_prob > config.SEG_THRESHOLD).astype(np.float32)

    # â”€â”€ Step 5: Grad-CAM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    gradcam_layer = get_gradcam_layer()
    gradcam_img = generate_gradcam_visualization(
        classifier,
        classifier_img,
        pred_class_idx=prediction["predicted_class"],
        last_conv_layer_name=gradcam_layer,
        display_image=clf_img,
    )

    # â”€â”€ Step 6: Save output images â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Segmentation mask
    seg_filename = f"seg_{scan_id}.png"
    seg_save_path = os.path.join(config.RESULTS_FOLDER, seg_filename)
    save_mask_as_image(seg_binary, seg_save_path)

    # Grad-CAM overlay
    gradcam_filename = f"gradcam_{scan_id}.png"
    gradcam_save_path = os.path.join(config.RESULTS_FOLDER, gradcam_filename)
    cv2.imwrite(gradcam_save_path, cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR))

    # Composite 4-panel result
    composite_filename = f"composite_{scan_id}.png"
    composite_save_path = os.path.join(config.RESULTS_FOLDER, composite_filename)
    create_result_composite(clf_img, seg_binary, gradcam_img, prediction, save_path=composite_save_path)

    # â”€â”€ Step 7: Build result dict â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Paths relative to app/static/ for use with url_for('static', filename=...)
    def to_static_rel(abs_path):
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        return os.path.relpath(abs_path, static_dir).replace("\\", "/")

    result = {
        "scan_id": scan_id,
        "original_filename": original_filename,
        # Classification
        "predicted_class":    prediction["predicted_class"],
        "predicted_label":    prediction["predicted_label"],
        "confidence":         prediction["confidence"],
        "all_probabilities":  json.dumps(prediction["all_probabilities"]),
        # File paths (relative to static/)
        "upload_path":        to_static_rel(image_path),
        "segmentation_path":  to_static_rel(seg_save_path),
        "gradcam_path":       to_static_rel(gradcam_save_path),
        "composite_path":     to_static_rel(composite_save_path),
    }

    return result
