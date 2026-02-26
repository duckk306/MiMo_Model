# visualize_phase1.py

import os
import cv2
import torch
import numpy as np

from models.MiMo import MiMo
from datasets.BDD100kDriveDataset import BDD100kDriveDataset


# ================= CONFIG =================
DATA_ROOT = "data/bdd100k"
CHECKPOINT = "checkpoints/mimo_phase1_best_loss.pth"
IMG_SIZE = 512
NUM_SAMPLES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "vis_phase1"
os.makedirs(SAVE_DIR, exist_ok=True)
# =========================================


def denormalize(img):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def main():
    print(">>> Visualizing Phase 1 Drivable Area")
    print(">>> Loading model")

    # -------- Model --------
    model = MiMo(num_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    # -------- Dataset --------
    dataset = BDD100kDriveDataset(
        root=DATA_ROOT,
        split="val",
        img_size=IMG_SIZE
    )

    # -------- Inference --------
    with torch.no_grad():
        for idx in range(NUM_SAMPLES):
            sample = dataset[idx]
            img = sample["image"].unsqueeze(0).to(DEVICE)

            out = model(img, mode="seg")
            mask = torch.sigmoid(out["drive_area"])[0, 0].cpu().numpy()

            # threshold
            mask_bin = (mask > 0.5).astype(np.uint8)

            # original image (denorm)
            img_np = sample["image"].permute(1, 2, 0).cpu().numpy()
            img_np = denormalize(img_np)

            # color overlay (green = drivable)
            overlay = img_np.copy()
            overlay[mask_bin == 1] = (
                0.5 * overlay[mask_bin == 1]
                + 0.5 * np.array([0, 255, 0])
            )

            # concat for comparison
            concat = np.concatenate([img_np, overlay], axis=1)

            save_path = os.path.join(SAVE_DIR, f"sample_{idx}.png")
            cv2.imwrite(save_path, cv2.cvtColor(concat, cv2.COLOR_RGB2BGR))

            print(f">>> Saved {save_path}")

    print("\n✅ Visualization completed")


if __name__ == "__main__":
    main()
