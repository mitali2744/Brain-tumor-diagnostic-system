"""
Model Evaluation Script
========================
Evaluates both trained models and prints a full metrics report:

Classification:
  - Accuracy, Precision, Recall, F1-Score (per class + macro/weighted)
  - Confusion Matrix

Segmentation:
  - Dice Score, IoU (Jaccard), Pixel Accuracy

Run:
    python evaluate.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.classification.efficientnet_model import load_classifier
from src.classification.dataset import build_classification_generators
from src.segmentation.unet import load_unet
from src.segmentation.dataset import build_segmentation_generators
from src.utils.visualization import plot_confusion_matrix


# ─── Classification Evaluation ────────────────────────────────────────────────

def evaluate_classifier():
    print("\n" + "=" * 60)
    print("  CLASSIFICATION MODEL EVALUATION")
    print("=" * 60)

    if not os.path.exists(config.CLASSIFIER_MODEL_PATH):
        print(f"❌ Classifier model not found at {config.CLASSIFIER_MODEL_PATH}")
        print("   Run: python train_classification.py first")
        return

    print("Loading classifier model...")
    model = load_classifier()

    print("Loading test data...")
    _, _, test_gen, _ = build_classification_generators()

    if test_gen is None:
        print("❌ No test data found. Check data/classification/Testing/ folder.")
        return

    # Collect predictions
    all_preds, all_labels = [], []
    for images, labels in test_gen:
        preds = model.predict(images, verbose=0)
        all_preds.extend(np.argmax(preds, axis=1))
        all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    target_names = [config.CLASS_LABELS[i] for i in range(config.NUM_CLASSES)]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(
        cm,
        class_names=config.CLASS_NAMES,
        save_path=os.path.join(config.MODELS_DIR, "confusion_matrix.png"),
    )
    print(f"✅ Confusion matrix saved to models/confusion_matrix.png")


# ─── Segmentation Evaluation ──────────────────────────────────────────────────

def dice_score_numpy(y_true: np.ndarray, y_pred: np.ndarray, smooth=1e-7) -> float:
    """Compute Dice score between two binary masks."""
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def iou_score(y_true: np.ndarray, y_pred: np.ndarray, smooth=1e-7) -> float:
    """Intersection over Union (Jaccard Index)."""
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def evaluate_segmentation():
    print("\n" + "=" * 60)
    print("  SEGMENTATION MODEL EVALUATION")
    print("=" * 60)

    if not os.path.exists(config.SEGMENTATION_MODEL_PATH):
        print(f"❌ Segmentation model not found at {config.SEGMENTATION_MODEL_PATH}")
        print("   Run: python train_segmentation.py first")
        return

    print("Loading U-Net model...")
    model = load_unet()

    print("Loading validation data...")
    _, val_gen = build_segmentation_generators()

    dice_scores, iou_scores, pixel_accs = [], [], []

    for images, masks in val_gen:
        preds = model.predict(images, verbose=0)
        preds_binary = (preds > config.SEG_THRESHOLD).astype(np.float32)

        for i in range(len(images)):
            true_mask = masks[i]
            pred_mask = preds_binary[i]

            dice_scores.append(dice_score_numpy(true_mask, pred_mask))
            iou_scores.append(iou_score(true_mask, pred_mask))

            # Pixel accuracy
            correct = np.sum(true_mask == pred_mask)
            total = true_mask.size
            pixel_accs.append(correct / total)

    print(f"\nDice Score  (mean ± std): {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"IoU Score   (mean ± std): {np.mean(iou_scores):.4f}  ± {np.std(iou_scores):.4f}")
    print(f"Pixel Acc   (mean ± std): {np.mean(pixel_accs):.4f}  ± {np.std(pixel_accs):.4f}")

    # Save Dice distribution plot
    plt.figure(figsize=(8, 4))
    plt.hist(dice_scores, bins=30, color="steelblue", edgecolor="white")
    plt.title("Dice Score Distribution (Validation Set)")
    plt.xlabel("Dice Score")
    plt.ylabel("Count")
    plt.axvline(np.mean(dice_scores), color="red", linestyle="--", label=f"Mean = {np.mean(dice_scores):.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.MODELS_DIR, "dice_distribution.png"), dpi=150)
    print(f"✅ Dice distribution plot saved to models/dice_distribution.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluate_classifier()
    evaluate_segmentation()
    print("\n✅ Evaluation complete.")
