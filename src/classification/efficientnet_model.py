"""
EfficientNetB0 Classifier for Brain Tumor Type
------------------------------------------------
Uses EfficientNetB0 pretrained on ImageNet as feature extractor.
Fine-tuned for 4-class classification:
  0: Glioma
  1: Meningioma
  2: No Tumor
  3: Pituitary

Strategy:
  Phase 1 — Train only the new classification head (base frozen)
  Phase 2 — Unfreeze top layers of EfficientNetB0 and fine-tune end-to-end
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def build_classifier(num_classes: int = config.NUM_CLASSES, dropout_rate: float = 0.3) -> Model:
    """
    Build EfficientNetB0-based classifier.

    Args:
        num_classes: Number of output classes (4)
        dropout_rate: Dropout before final Dense layer

    Returns:
        Keras Model (head unfrozen, base frozen — ready for Phase 1 training)
    """
    # Load EfficientNetB0 without top classification layer
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
    )
    base_model.trainable = False  # Freeze base for Phase 1

    inputs = layers.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS), name="mri_input")

    # The application model includes its own ImageNet preprocessing and expects
    # pixels in [0, 255]. Project preprocessing emits [0, 1], so restore the
    # expected range before passing inputs to EfficientNetB0.
    scaled_inputs = layers.Rescaling(255.0, name="restore_pixel_scale")(inputs)

    # training=False keeps BatchNorm in inference mode during Phase 1
    x = base_model(scaled_inputs, training=False)
    # Expose a connected top-level feature tensor for Grad-CAM. Accessing an
    # internal tensor from a nested Keras application model is not graph-safe.
    x = layers.Activation("linear", name="gradcam_features")(x)

    # Classification head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name="EfficientNetB0_BrainTumorClassifier")
    return model, base_model


def compile_phase1(model: Model) -> Model:
    """
    Compile for Phase 1: train only the classification head.
    Higher LR is fine since base is frozen.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_top_layers(base_model, num_layers_to_unfreeze: int = 20):
    """
    Unfreeze the last N layers of EfficientNetB0 for Phase 2 fine-tuning.
    Only unfreezes BatchNorm-safe layers (Conv, Depthwise, etc.).
    """
    base_model.trainable = True

    # Freeze all layers except the last `num_layers_to_unfreeze`
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    # Keep BatchNorm layers frozen to avoid instability
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def compile_phase2(model: Model) -> Model:
    """
    Compile for Phase 2: fine-tuning unfrozen layers.
    Use a much lower LR to avoid destroying pretrained weights.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_classifier(model_path: str = config.CLASSIFIER_MODEL_PATH) -> Model:
    """Load a saved classifier model."""
    return tf.keras.models.load_model(model_path)


def predict_tumor_type(model: Model, preprocessed_img: "np.ndarray") -> dict:
    """
    Run inference on a single preprocessed image.

    Args:
        model: Loaded classifier model
        preprocessed_img: Float32 array (224, 224, 3), values in [0,1]

    Returns:
        dict with keys:
          - predicted_class: int (0-3)
          - predicted_label: str (e.g. "Glioma")
          - confidence: float (0-100)
          - all_probabilities: dict {label: probability}
    """
    import numpy as np

    img_batch = np.expand_dims(preprocessed_img, axis=0)   # (1, 224, 224, 3)
    probs = model.predict(img_batch, verbose=0)[0]           # (4,)

    predicted_class = int(np.argmax(probs))
    predicted_label = config.CLASS_LABELS[predicted_class]
    confidence = float(probs[predicted_class]) * 100

    all_probabilities = {
        config.CLASS_LABELS[i]: round(float(probs[i]) * 100, 2)
        for i in range(len(probs))
    }

    return {
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "confidence": round(confidence, 2),
        "all_probabilities": all_probabilities,
    }
