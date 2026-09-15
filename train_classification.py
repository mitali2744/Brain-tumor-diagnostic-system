"""
Train EfficientNetB0 Classifier for Brain Tumor Classification
==============================================================
Two-phase training strategy:
  Phase 1 (5 epochs)  — freeze EfficientNetB0 base, train only classification head
  Phase 2 (25 epochs) — unfreeze top 20 layers and fine-tune end-to-end

Run:
    python train_classification.py
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Make sure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.classification.efficientnet_model import (
    build_classifier,
    compile_phase1,
    compile_phase2,
    unfreeze_top_layers,
)
from src.classification.dataset import build_classification_generators
from src.utils.visualization import plot_training_history

os.makedirs(config.MODELS_DIR, exist_ok=True)


def get_callbacks(phase: int):
    """
    Keras callbacks used during training.
    - EarlyStopping: stops if val_loss doesn't improve
    - ModelCheckpoint: saves best weights
    - ReduceLROnPlateau: halves LR when plateau is detected
    """
    checkpoint_path = os.path.join(config.MODELS_DIR, f"classifier_phase{phase}_best.h5")
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def main():
    print("=" * 60)
    print("  Brain Tumor Classifier — Training")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────
    print("\n[1/4] Loading dataset...")
    train_gen, val_gen, test_gen, class_weights = build_classification_generators()

    # ── Build model ────────────────────────────────────────────────
    print("\n[2/4] Building EfficientNetB0 model...")
    model, base_model = build_classifier()
    model.summary()

    # ── Phase 1: Train head only ───────────────────────────────────
    print("\n[3/4] Phase 1 — Training classification head (base frozen)...")
    model = compile_phase1(model)
    history_p1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=5,
        class_weight=class_weights,
        callbacks=get_callbacks(phase=1),
        verbose=1,
    )
    plot_training_history(
        history_p1,
        title="Phase 1 — Head Training",
        save_path=os.path.join(config.MODELS_DIR, "classifier_phase1_history.png"),
    )

    # ── Phase 2: Fine-tune top layers ──────────────────────────────
    print("\n[4/4] Phase 2 — Fine-tuning top 20 layers of EfficientNetB0...")
    unfreeze_top_layers(base_model, num_layers_to_unfreeze=20)
    model = compile_phase2(model)
    history_p2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS_CLASSIFIER,
        class_weight=class_weights,
        callbacks=get_callbacks(phase=2),
        verbose=1,
    )
    plot_training_history(
        history_p2,
        title="Phase 2 — Fine-tuning",
        save_path=os.path.join(config.MODELS_DIR, "classifier_phase2_history.png"),
    )

    # ── Save final model ───────────────────────────────────────────
    model.save(config.CLASSIFIER_MODEL_PATH)
    print(f"\n✅ Classifier saved to: {config.CLASSIFIER_MODEL_PATH}")

    # ── Quick test evaluation ──────────────────────────────────────
    if test_gen is not None:
        print("\n--- Test Set Evaluation ---")
        results = model.evaluate(test_gen, verbose=1)
        print(f"Test Loss: {results[0]:.4f} | Test Accuracy: {results[1]:.4f}")


if __name__ == "__main__":
    main()
