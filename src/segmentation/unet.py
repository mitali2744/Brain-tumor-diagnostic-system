"""
U-Net Architecture for Brain Tumor Segmentation
-------------------------------------------------
Standard U-Net with:
  - Encoder (contracting path): 4 downsampling blocks
  - Bottleneck
  - Decoder (expanding path): 4 upsampling blocks with skip connections
  - Output: sigmoid activation → binary mask

Input:  (batch, 256, 256, 3)
Output: (batch, 256, 256, 1)  — pixel-wise tumor probability
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def conv_block(x, filters: int, dropout_rate: float = 0.0):
    """
    Double convolution block: Conv2D → BN → ReLU → Conv2D → BN → ReLU
    Standard U-Net building block.
    """
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def encoder_block(x, filters: int, dropout_rate: float = 0.0):
    """Conv block + MaxPool. Returns (pooled_output, skip_connection)."""
    skip = conv_block(x, filters, dropout_rate)
    pooled = layers.MaxPooling2D(2)(skip)
    return pooled, skip


def decoder_block(x, skip, filters: int):
    """
    Upsample + concatenate skip connection + double conv.
    Uses Conv2DTranspose for learnable upsampling.
    """
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x


def build_unet(
    input_shape=(config.SEG_IMG_HEIGHT, config.SEG_IMG_WIDTH, config.SEG_IMG_CHANNELS),
    base_filters: int = 64,
) -> Model:
    """
    Build and return a compiled U-Net model.

    Args:
        input_shape: (H, W, C) — default 256×256×3
        base_filters: Number of filters in first conv block (doubles each level)

    Returns:
        Compiled Keras Model
    """
    inputs = layers.Input(shape=input_shape, name="mri_input")

    # ── Encoder ──────────────────────────────────────────────────────────────
    e1, s1 = encoder_block(inputs, base_filters * 1)         # 256→128, 64 filters
    e2, s2 = encoder_block(e1, base_filters * 2)             # 128→64,  128 filters
    e3, s3 = encoder_block(e2, base_filters * 4)             # 64→32,   256 filters
    e4, s4 = encoder_block(e3, base_filters * 8, dropout_rate=0.3)  # 32→16, 512 filters

    # ── Bottleneck ───────────────────────────────────────────────────────────
    bottleneck = conv_block(e4, base_filters * 16, dropout_rate=0.3)  # 16×16, 1024

    # ── Decoder ──────────────────────────────────────────────────────────────
    d4 = decoder_block(bottleneck, s4, base_filters * 8)     # 16→32,  512
    d3 = decoder_block(d4, s3, base_filters * 4)             # 32→64,  256
    d2 = decoder_block(d3, s2, base_filters * 2)             # 64→128, 128
    d1 = decoder_block(d2, s1, base_filters * 1)             # 128→256, 64

    # ── Output ───────────────────────────────────────────────────────────────
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="segmentation_output")(d1)

    model = Model(inputs, outputs, name="U-Net")
    return model


def dice_coefficient(y_true, y_pred, smooth: float = 1e-7):
    """
    Dice coefficient metric.
    Measures overlap between predicted and ground truth masks.
    Range: 0 (no overlap) to 1 (perfect overlap).
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    """Dice loss = 1 - Dice coefficient. Used as training loss."""
    return 1.0 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    """
    Binary Cross-Entropy + Dice Loss.
    BCE handles pixel-level probabilities; Dice handles class imbalance.
    Best practice for medical image segmentation.
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


def get_compiled_unet() -> Model:
    """Build and compile the U-Net with combined loss and metrics."""
    model = build_unet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=combined_loss,
        metrics=[dice_coefficient, "accuracy"],
    )
    return model


def load_unet(model_path: str = config.SEGMENTATION_MODEL_PATH) -> Model:
    """Load a saved U-Net model with custom loss/metric functions."""
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "combined_loss": combined_loss,
            "dice_coefficient": dice_coefficient,
            "dice_loss": dice_loss,
        },
    )
