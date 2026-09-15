"""
Classification Dataset Generator
-----------------------------------
Loads the Kaggle Brain Tumor MRI Dataset.
Expected folder structure:
  data/classification/
    Training/
      glioma/        *.jpg
      meningioma/    *.jpg
      notumor/       *.jpg
      pituitary/     *.jpg
    Testing/
      glioma/
      ...

Returns Keras generators or numpy arrays for training/validation/testing.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.preprocessing.preprocess import (
    preprocess_for_classification,
    get_classification_augmentor,
    apply_classification_augmentation,
)


CLASS_NAMES = config.CLASS_NAMES   # ["glioma", "meningioma", "notumor", "pituitary"]


def scan_dataset(data_dir: str):
    """
    Scan a folder of class subfolders and return (image_paths, labels).

    Args:
        data_dir: Path containing one subfolder per class

    Returns:
        image_paths: list of absolute file paths
        labels: list of integer class indices
    """
    image_paths, labels = [], []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: class folder not found: {class_dir}")
            continue

        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(class_idx)

    print(f"Scanned {data_dir}: {len(image_paths)} images across {len(set(labels))} classes")
    return image_paths, labels


def compute_weights(labels):
    """Compute per-class weights to handle class imbalance."""
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return dict(zip(classes, weights))


class ClassificationDataGenerator(tf.keras.utils.Sequence):
    """
    Keras Sequence for classification data.
    Loads and preprocesses images on-the-fly to avoid memory issues.
    """

    def __init__(self, image_paths, labels, batch_size=config.BATCH_SIZE, augment=False, shuffle=True):
        self.image_paths = image_paths
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.augmentor = get_classification_augmentor() if augment else None
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return max(1, len(self.image_paths) // self.batch_size)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]

        images, labels = [], []
        for i in batch_indices:
            try:
                img = preprocess_for_classification(self.image_paths[i])
                if self.augment:
                    img = apply_classification_augmentation(img, self.augmentor)
                images.append(img)
                labels.append(self.labels[i])
            except Exception as e:
                print(f"Warning: skipping {self.image_paths[i]} — {e}")

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def build_classification_generators(
    train_dir: str = None,
    test_dir: str = None,
    data_dir: str = config.CLASSIFICATION_DATA_DIR,
):
    """
    Build train, validation, and test generators.

    Returns:
        train_gen, val_gen, test_gen, class_weights
    """
    if train_dir is None:
        train_dir = os.path.join(data_dir, "Training")
    if test_dir is None:
        test_dir = os.path.join(data_dir, "Testing")

    # Load training data
    train_paths, train_labels = scan_dataset(train_dir)
    if not train_paths:
        raise ValueError(f"No training images found in {train_dir}")

    # Split training into train + validation
    tr_paths, val_paths, tr_labels, val_labels = train_test_split(
        train_paths, train_labels,
        test_size=config.VALIDATION_SPLIT,
        stratify=train_labels,
        random_state=42,
    )

    # Load test data if available
    test_gen = None
    if os.path.isdir(test_dir):
        test_paths, test_labels = scan_dataset(test_dir)
        test_gen = ClassificationDataGenerator(test_paths, test_labels, augment=False, shuffle=False)

    class_weights = compute_weights(tr_labels)

    train_gen = ClassificationDataGenerator(tr_paths, tr_labels, augment=True, shuffle=True)
    val_gen = ClassificationDataGenerator(val_paths, val_labels, augment=False, shuffle=False)

    print(f"Train: {len(tr_paths)} | Val: {len(val_paths)}")
    print(f"Class weights: {class_weights}")

    return train_gen, val_gen, test_gen, class_weights
