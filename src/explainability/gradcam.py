"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
-------------------------------------------------------
Manual implementation — no external libraries required.

How it works:
  1. Forward pass → get feature maps from the last convolutional layer
  2. Backward pass → get gradients of predicted class score w.r.t. those feature maps
  3. Pool gradients spatially (global average) → importance weights per channel
  4. Weighted sum of feature maps → raw CAM
  5. ReLU → only keep positive activations
  6. Resize to input image size
  7. Normalize and overlay as heatmap on original image

Reference: Selvaraju et al. 2017 — "Grad-CAM: Visual Explanations from Deep Networks"
"""

import numpy as np
import cv2
import tensorflow as tf
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def get_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    pred_class_idx: int = None,
    last_conv_layer_name: str = config.GRADCAM_LAYER,
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for a given image and class.

    Args:
        model: Trained EfficientNetB0 classifier
        img_array: Preprocessed image, shape (224, 224, 3), float32 in [0,1]
        pred_class_idx: Class index to explain. If None, uses the top predicted class.
        last_conv_layer_name: Name of the last Conv layer to hook into.

    Returns:
        heatmap: float32 array (224, 224), values in [0, 1]
    """
    # Build a sub-model that outputs:
    #   - feature maps from last conv layer
    #   - final predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    img_batch = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch, training=False)
        if pred_class_idx is None:
            pred_class_idx = tf.argmax(predictions[0]).numpy()
        class_score = predictions[:, pred_class_idx]  # scalar score for target class

    # Gradients of class score w.r.t. conv feature maps
    grads = tape.gradient(class_score, conv_outputs)  # shape: (1, h, w, C)

    # Global average pooling of gradients → importance weights per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape: (C,)

    # Weighted sum of feature maps
    conv_outputs = conv_outputs[0]                        # (h, w, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
    heatmap = tf.squeeze(heatmap)                         # (h, w)

    # ReLU — only keep features that positively influence the class
    heatmap = tf.nn.relu(heatmap).numpy()

    # Normalize to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Resize to original image dimensions
    heatmap = cv2.resize(heatmap, (config.IMG_WIDTH, config.IMG_HEIGHT))
    return heatmap.astype(np.float32)


def overlay_heatmap_on_image(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = config.GRADCAM_ALPHA,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        original_img: RGB float32 (224, 224, 3) in [0,1] OR uint8 in [0,255]
        heatmap: float32 (224, 224) in [0, 1]
        alpha: Blend weight for the heatmap (0=no heatmap, 1=only heatmap)
        colormap: OpenCV colormap for heatmap visualization

    Returns:
        overlaid: uint8 RGB image (224, 224, 3)
    """
    # Convert original to uint8 if needed
    if original_img.dtype == np.float32 or original_img.max() <= 1.0:
        original_uint8 = (original_img * 255).astype(np.uint8)
    else:
        original_uint8 = original_img.astype(np.uint8)

    # Convert heatmap to uint8 and apply colormap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)  # BGR
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # Blend: (1 - alpha) * original + alpha * heatmap
    overlaid = cv2.addWeighted(original_uint8, 1 - alpha, colored_heatmap, alpha, 0)
    return overlaid


def generate_gradcam_visualization(
    model: tf.keras.Model,
    img_array: np.ndarray,
    pred_class_idx: int = None,
    save_path: str = None,
) -> np.ndarray:
    """
    End-to-end Grad-CAM: compute heatmap and overlay on image.

    Args:
        model: Trained classifier
        img_array: Preprocessed image float32 (224, 224, 3) in [0,1]
        pred_class_idx: Target class. None → top predicted class.
        save_path: If provided, saves the overlaid image to this path.

    Returns:
        overlaid: uint8 RGB image with heatmap overlay (224, 224, 3)
    """
    heatmap = get_gradcam_heatmap(model, img_array, pred_class_idx)
    overlaid = overlay_heatmap_on_image(img_array, heatmap)

    if save_path:
        # Save as BGR for OpenCV
        cv2.imwrite(save_path, cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))

    return overlaid


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """
    Auto-detect the name of the last convolutional layer in the model.
    Useful when the default layer name doesn't match.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # Prefer a connected top-level spatial feature layer exposed by a
        # nested application backbone (e.g. EfficientNet).
        if len(layer.output.shape) == 4 and layer.name == config.GRADCAM_LAYER:
            return layer.name
    raise ValueError("No Conv2D layer found in model.")
