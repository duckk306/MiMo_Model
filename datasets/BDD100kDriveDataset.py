# datasets/BDD100kDriveDataset.py (PATH-CORRECTED)

import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class BDD100kDriveDataset(Dataset):
    """
    Phase 1: Drivable-area segmentation
    """

    def __init__(self, root, split="train", img_size=640):
        self.img_dir = os.path.join(
            root, "images100k", "100k", split
        )
        self.lbl_dir = os.path.join(
            root, "labels", "drivable", "labels", split
        )
        self.img_size = img_size

        self.images = sorted(os.listdir(self.img_dir))
        self.labels = sorted(os.listdir(self.lbl_dir))
        assert len(self.images) == len(self.labels)

        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_dir, self.images[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img).permute(2, 0, 1)

        lbl = cv2.imread(
            os.path.join(self.lbl_dir, self.labels[idx]),
            cv2.IMREAD_GRAYSCALE
        )
        lbl = cv2.resize(
            lbl, (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST
        )

        drive = ((lbl == 1) | (lbl == 2)).astype(np.float32)
        drive = torch.from_numpy(drive).unsqueeze(0)

        return {
            "image": img,
            "drive_area": drive
        }
