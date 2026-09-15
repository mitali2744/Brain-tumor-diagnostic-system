"""
Train U-Net for Brain Tumor Segmentation
=========================================
Trains the custom U-Net on the LGG MRI Segmentation dataset.
Loss: Binary Cross-Entropy + Dice Loss (combined)
Metric: Dice Coefficient

Run:
    python train_segmentation.py
"""

import os
import sys
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.segmentation.unet import get_compiled_unet, load_unet
from src.segmentation.dataset import build_segmentation_generators
from src.utils.visualization import plot_training_history

os.makedirs(config.MODELS_DIR, exist_ok=True)


def get_callbacks():
    checkpoint_path = os.path.join(config.MODELS_DIR, "unet_best.h5")
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coefficient",
            patience=8,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_dice_coefficient",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_dice_coefficient",
            factor=0.5,
            patience=4,
            mode="max",
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def main():
    print("=" * 60)
    print("  Brain Tumor Segmentation — U-Net Training")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────
    print("\n[1/3] Loading segmentation dataset...")
    train_gen, val_gen = build_segmentation_generators()

    # ── Build model ────────────────────────────────────────────────
    print("\n[2/3] Building U-Net model...")
    model = get_compiled_unet()
    model.summary()

    # ── Train ──────────────────────────────────────────────────────
    print("\n[3/3] Training U-Net...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS_SEGMENTATION,
        callbacks=get_callbacks(),
        verbose=1,
    )

    plot_training_history(
        history,
        title="U-Net Segmentation Training",
        save_path=os.path.join(config.MODELS_DIR, "unet_training_history.png"),
    )

    # ── Save final model ───────────────────────────────────────────
    model.save(config.SEGMENTATION_MODEL_PATH)
    print(f"\n✅ U-Net model saved to: {config.SEGMENTATION_MODEL_PATH}")

    # ── Evaluate on validation set ─────────────────────────────────
    print("\n--- Validation Set Evaluation ---")
    results = model.evaluate(val_gen, verbose=1)
    metric_names = model.metrics_names
    for name, val in zip(metric_names, results):
        print(f"  {name}: {val:.4f}")


if __name__ == "__main__":
    main()
