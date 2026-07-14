"""
RSNA Pneumonia Detection dataset (PyTorch Dataset), translated from
data_loader/RSNA_dataloader.py.

Expected layout (standard Kaggle layout):
    train_path/<patientId>.dcm
    csv_path -> stage_2_train_labels.csv with columns:
        patientId, x, y, width, height, Target
    (rows with Target==0 have NaN box columns)
"""

import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset

from .. import config
from ..utils.bbox_utils import boxes_to_yolo_target


def load_rsna_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    return df


def get_default_train_augmentation():
    return A.Compose([
        A.OneOf([A.RandomGamma(), A.RandomBrightnessContrast()], p=0.3),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"],
                                 min_visibility=0.3))


class RSNADataset(Dataset):
    def __init__(self, df, patient_ids, img_dir, dim=(config.IMG_SIZE, config.IMG_SIZE),
                 augmentation=None, normalize=True):
        self.df = df
        self.patient_ids = list(patient_ids)
        self.img_dir = img_dir
        self.dim = dim
        self.augmentation = augmentation
        self.normalize = normalize

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        rows = self.df[self.df["patientId"] == patient_id]

        img_path = os.path.join(self.img_dir, patient_id + ".dcm")
        dcm = pydicom.dcmread(img_path)
        img = dcm.pixel_array
        img = cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
        orig_h, orig_w = img.shape[:2]

        boxes = []
        if rows["Target"].iloc[0] == 1:
            for _, row in rows.iterrows():
                x1, y1 = float(row["x"]), float(row["y"])
                x2, y2 = x1 + float(row["width"]), y1 + float(row["height"])
                # rescale to network input size
                x1 = x1 / orig_w * self.dim[0]
                x2 = x2 / orig_w * self.dim[0]
                y1 = y1 / orig_h * self.dim[1]
                y2 = y2 / orig_h * self.dim[1]
                boxes.append([x1, y1, x2, y2])

        img = cv2.resize(img, self.dim)

        if self.augmentation is not None and len(boxes) > 0:
            augmented = self.augmentation(image=img, bboxes=boxes,
                                           class_labels=[0] * len(boxes))
            img = augmented["image"]
            boxes = augmented["bboxes"]
        elif self.augmentation is not None:
            augmented = self.augmentation(image=img, bboxes=[], class_labels=[])
            img = augmented["image"]

        img = img.astype(np.float32)
        if self.normalize:
            img = img / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1).float()

        target = boxes_to_yolo_target(np.array(boxes) if len(boxes) else np.zeros((0, 4)))
        target_t = torch.from_numpy(target).float()

        class_label = 2 if len(boxes) > 0 else 0     # 2 = pneumonia
        return img_t, target_t, class_label, boxes


def train_val_split(df, val_fraction=0.2, seed=config.SEED):
    ids = df["patientId"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_fraction)
    return ids[n_val:], ids[:n_val]


def rsna_collate_fn(batch):
    imgs, targets, class_labels, boxes_list = zip(*batch)
    imgs = torch.stack(imgs)
    targets = torch.stack(targets)
    class_labels = torch.tensor(class_labels)
    return imgs, targets, class_labels, list(boxes_list)
