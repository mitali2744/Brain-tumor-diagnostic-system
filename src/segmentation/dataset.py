"""
Segmentation Dataset Generator
--------------------------------
Loads LGG MRI Segmentation dataset (BraTS-style).
Expected folder structure:
  data/segmentation/
    ├── <patient_folder>/
    │   ├── <slice>_mask.tif   ← ground truth mask
    │   └── <slice>.tif        ← MRI slice
    ...

Yields (image, mask) batches for U-Net training.
"""

import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.preprocessing.preprocess import (
    preprocess_for_segmentation,
    get_segmentation_augmentor,
    apply_segmentation_augmentation,
)


def get_image_mask_pairs(data_dir: str):
    """
    Scan the LGG dataset directory and return paired (image_path, mask_path) lists.
    Masks are identified by the '_mask' suffix.
    """
    mask_paths = sorted(glob.glob(os.path.join(data_dir, "**", "*_mask.tif"), recursive=True))
    image_paths = [p.replace("_mask.tif", ".tif") for p in mask_paths]

    # Filter out pairs where the image file doesn't exist
    valid_pairs = [(img, msk) for img, msk in zip(image_paths, mask_paths) if os.path.exists(img)]

    print(f"Found {len(valid_pairs)} valid image-mask pairs in {data_dir}")
    return valid_pairs


def split_dataset(pairs, val_split=config.VALIDATION_SPLIT, seed=42):
    """Split (image, mask) pairs into train/validation sets."""
    train_pairs, val_pairs = train_test_split(pairs, test_size=val_split, random_state=seed)
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")
    return train_pairs, val_pairs


class SegmentationDataGenerator(tf.keras.utils.Sequence):
    """
    Keras Sequence generator for segmentation data.
    Supports augmentation during training.
    """

    def __init__(self, pairs, batch_size=config.BATCH_SIZE, augment=False, shuffle=True):
        self.pairs = pairs
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.augmentor = get_segmentation_augmentor() if augment else None
        self.indices = np.arange(len(self.pairs))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return max(1, len(self.pairs) // self.batch_size)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_pairs = [self.pairs[i] for i in batch_indices]

        images, masks = [], []
        for img_path, msk_path in batch_pairs:
            try:
                img, msk = preprocess_for_segmentation(img_path, msk_path)
                if self.augment:
                    img, msk = apply_segmentation_augmentation(img, msk, self.augmentor)
                images.append(img)
                masks.append(msk)
            except Exception as e:
                print(f"Warning: skipping {img_path} — {e}")

        return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def build_segmentation_generators(data_dir: str = config.SEGMENTATION_DATA_DIR):
    """
    Build train and validation generators for U-Net training.

    Returns:
        train_gen, val_gen
    """
    pairs = get_image_mask_pairs(data_dir)
    if len(pairs) == 0:
        raise ValueError(f"No image-mask pairs found in {data_dir}. Check dataset path.")

    train_pairs, val_pairs = split_dataset(pairs)
    train_gen = SegmentationDataGenerator(train_pairs, augment=True, shuffle=True)
    val_gen = SegmentationDataGenerator(val_pairs, augment=False, shuffle=False)
    return train_gen, val_gen
