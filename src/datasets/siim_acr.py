"""
SIIM-ACR Pneumothorax segmentation dataset (PyTorch Dataset), translated
from data_loader/SIIM_ACR_dataloader.py.

Expected layout (this is the standard Kaggle layout):
    train_path/
        */*/*.dcm                       (any nested folder structure)
    csv_path -> a csv with columns "ImageId", " EncodedPixels" (note the
                leading space, exactly as distributed by SIIM/Kaggle)

Each image may appear on several rows (one row per separate RLE region);
they are combined into a single binary mask per image.
"""

import os
from glob import glob

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset

from .. import config
from ..utils.mask_utils import masks_as_image


def _index_dicom_files(train_path):
    paths = glob(os.path.join(train_path, "**", "*.dcm"), recursive=True)
    return {os.path.basename(p).split(".dcm")[0]: p for p in paths}


def build_siim_dataframe(csv_path, train_path, only_positive=False):
    """only_positive: matches the upstream reference repo's default
    (coursat-ai/MultiCheXNet, SIIM_ACR_dataloader.get_train_validation_generator,
    only_positive=True by default). Drops every image with an entirely empty
    pneumothorax mask before training/val split.

    This is the canonical fix for the "predict nothing" collapse: with plain
    Dice loss (or even BCE+Dice), training on a mix where ~80% of masks are
    empty gives the model a trivial near-perfect-scoring shortcut. If the
    model never sees an empty mask during segmentation training, that
    shortcut doesn't exist, and Dice loss behaves as intended. This is
    stronger than the BCEDiceLoss + WeightedRandomSampler mitigation and is
    what the original authors ship as their default - use both together.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]           # normalize " EncodedPixels"
    file_map = _index_dicom_files(train_path)
    df["full_path"] = df["ImageId"].map(file_map.get)
    df = df.dropna(subset=["full_path"]).reset_index(drop=True)
    if only_positive:
        df = df[df["EncodedPixels"].str.strip() != "-1"].reset_index(drop=True)
    return df


def get_default_train_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([A.RandomGamma(), A.RandomBrightnessContrast()], p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05),
            A.GridDistortion(),
            A.OpticalDistortion(distort_limit=2),
        ], p=0.3),
        A.Rotate(limit=15, p=0.2),
    ])


class SIIMACRDataset(Dataset):
    def __init__(self, df, image_ids, dim=(config.IMG_SIZE, config.IMG_SIZE),
                 augmentation=None, normalize=True):
        """
        df         : dataframe returned by build_siim_dataframe
        image_ids  : list/array of ImageId values to include (train/val split)
        augmentation: an albumentations Compose or None
        """
        self.df = df
        self.image_ids = list(image_ids)
        self.dim = dim
        self.augmentation = augmentation
        self.normalize = normalize

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        rows = self.df[self.df["ImageId"] == image_id]

        dcm = pydicom.dcmread(rows["full_path"].iloc[0])
        img = dcm.pixel_array
        img = cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
        orig_h, orig_w = img.shape[:2]

        if (rows["EncodedPixels"].iloc[0]).strip() == "-1":
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        else:
            mask = masks_as_image(rows["EncodedPixels"].tolist(), (orig_w, orig_h)).T

        img = cv2.resize(img, self.dim)
        mask = cv2.resize(mask, self.dim, interpolation=cv2.INTER_NEAREST)

        if self.augmentation is not None:
            augmented = self.augmentation(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        img = img.astype(np.float32)
        if self.normalize:
            img = img / 255.0

        img_t = torch.from_numpy(img).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        mask_t = (mask_t > 0).float()

        # derived 3-class label: 0 normal, 1 pneumothorax, 2 pneumonia (n/a here)
        class_label = 1 if mask_t.sum() > 0 else 0
        return img_t, mask_t, class_label


def train_val_split(df, val_fraction=0.2, seed=config.SEED):
    ids = df["ImageId"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_fraction)
    return ids[n_val:], ids[:n_val]


def compute_pos_neg_sample_weights(df, image_ids):
    """Per-image weights (inverse class frequency) for a WeightedRandomSampler.

    Only ~20% of SIIM-ACR images actually contain a pneumothorax mask, so a
    plain `shuffle=True` DataLoader shows the segmenter mostly-empty masks
    and reinforces the "predict nothing" collapse. Sampling with these
    weights (replacement=True) roughly balances positive/negative images
    within every batch instead.

    Returns a torch.DoubleTensor the same length/order as `image_ids`,
    ready to pass to `torch.utils.data.WeightedRandomSampler`.
    """
    first_rows = df.drop_duplicates(subset="ImageId").set_index("ImageId")
    is_positive = np.array([
        first_rows.loc[image_id, "EncodedPixels"].strip() != "-1"
        for image_id in image_ids
    ])
    n_pos = int(is_positive.sum())
    n_neg = len(is_positive) - n_pos
    weights = np.where(is_positive, 1.0 / max(n_pos, 1), 1.0 / max(n_neg, 1))
    return torch.as_tensor(weights, dtype=torch.double)
