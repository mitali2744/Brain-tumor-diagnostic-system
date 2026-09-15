"""
Visualization Utilities
------------------------
Plotting helpers for training curves, segmentation overlays,
confusion matrices, and result dashboards.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def plot_training_history(history, title: str = "Training History", save_path: str = None):
    """
    Plot training and validation accuracy + loss curves.

    Args:
        history: Keras History object returned by model.fit()
        title: Plot title
        save_path: If provided, saves the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14)

    # Accuracy
    axes[0].plot(history.history.get("accuracy", []), label="Train Accuracy", color="steelblue")
    axes[0].plot(history.history.get("val_accuracy", []), label="Val Accuracy", color="tomato")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history.get("loss", []), label="Train Loss", color="steelblue")
    axes[1].plot(history.history.get("val_loss", []), label="Val Loss", color="tomato")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_segmentation_result(original, mask_pred, mask_true=None, save_path: str = None):
    """
    Visualize segmentation result: original MRI + predicted mask + optional ground truth.

    Args:
        original: float32 (H, W, 3) image
        mask_pred: float32 (H, W) or (H, W, 1) predicted mask
        mask_true: float32 (H, W) or (H, W, 1) ground truth mask (optional)
        save_path: Path to save figure
    """
    mask_pred = np.squeeze(mask_pred)
    binary_pred = (mask_pred > config.SEG_THRESHOLD).astype(np.uint8)

    n_plots = 3 if mask_true is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original MRI")
    axes[0].axis("off")

    # Overlay predicted mask on original
    overlay = original.copy()
    if overlay.max() <= 1.0:
        overlay = (overlay * 255).astype(np.uint8)
    overlay[binary_pred == 1] = [255, 0, 0]   # Red = tumor
    axes[1].imshow(overlay)
    axes[1].set_title("Predicted Segmentation")
    axes[1].axis("off")

    if mask_true is not None:
        mask_true = np.squeeze(mask_true)
        axes[2].imshow(mask_true, cmap="Reds")
        axes[2].set_title("Ground Truth Mask")
        axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names=config.CLASS_NAMES, save_path: str = None):
    """
    Plot a normalized confusion matrix.

    Args:
        cm: Confusion matrix from sklearn
        class_names: List of class label strings
        save_path: Path to save figure
    """
    import seaborn as sns

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[c.capitalize() for c in class_names],
        yticklabels=[c.capitalize() for c in class_names],
        ax=ax,
    )
    ax.set_title("Normalized Confusion Matrix", fontsize=13)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def create_result_composite(original_img, seg_mask, gradcam_img, prediction: dict, save_path: str = None):
    """
    Create a 4-panel composite for a single scan result:
      [Original MRI] [Segmentation Overlay] [Grad-CAM] [Bar Chart of probabilities]

    Args:
        original_img: float32 or uint8 RGB (224, 224, 3)
        seg_mask: float32 (256, 256, 1) or (256, 256) — will be resized
        gradcam_img: uint8 RGB (224, 224, 3)
        prediction: dict from predict_tumor_type()
        save_path: Path to save composite

    Returns:
        fig: matplotlib Figure
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Panel 1: Original MRI
    if original_img.max() <= 1.0:
        original_uint8 = (original_img * 255).astype(np.uint8)
    else:
        original_uint8 = original_img.astype(np.uint8)
    axes[0].imshow(original_uint8)
    axes[0].set_title("Original MRI Scan", fontsize=11)
    axes[0].axis("off")

    # Panel 2: Segmentation overlay
    mask = np.squeeze(seg_mask)
    mask_resized = cv2.resize(mask, (config.IMG_WIDTH, config.IMG_HEIGHT))
    binary_mask = (mask_resized > config.SEG_THRESHOLD).astype(np.uint8)
    seg_overlay = original_uint8.copy()
    seg_overlay[binary_mask == 1] = [255, 50, 50]   # Red highlight
    axes[1].imshow(seg_overlay)
    axes[1].set_title("U-Net Segmentation", fontsize=11)
    axes[1].axis("off")

    # Panel 3: Grad-CAM
    axes[2].imshow(gradcam_img)
    axes[2].set_title("Grad-CAM Explanation", fontsize=11)
    axes[2].axis("off")

    # Panel 4: Probability bar chart
    labels = list(prediction["all_probabilities"].keys())
    probs = list(prediction["all_probabilities"].values())
    colors = ["#e74c3c" if l == prediction["predicted_label"] else "#3498db" for l in labels]
    bars = axes[3].barh(labels, probs, color=colors)
    axes[3].set_xlim(0, 100)
    axes[3].set_xlabel("Confidence (%)")
    axes[3].set_title(
        f"Prediction: {prediction['predicted_label']}\n({prediction['confidence']:.1f}% confidence)",
        fontsize=11,
    )
    for bar, prob in zip(bars, probs):
        axes[3].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f"{prob:.1f}%", va="center", fontsize=9)
    axes[3].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def save_mask_as_image(mask: np.ndarray, save_path: str):
    """Save a binary mask as a grayscale PNG."""
    mask_squeezed = np.squeeze(mask)
    mask_uint8 = (mask_squeezed * 255).astype(np.uint8)
    cv2.imwrite(save_path, mask_uint8)
