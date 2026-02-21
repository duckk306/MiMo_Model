# train_phase1_fixed.py

import os
import time
import cv2
import torch
from torch.utils.data import DataLoader

from models.MiMo import MiMo
from datasets.BDD100kDriveDataset import BDD100kDriveDataset
from losses.LossModules import SegmentationLoss


# ======================= CRITICAL FIXES =======================
# OpenCV + Windows + PyTorch fix (BẮT BUỘC)
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
# =============================================================


# ========================= CONFIG =============================
DATA_ROOT = "data/bdd100k"
IMG_SIZE = 512
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-3

NUM_WORKERS = 0
PIN_MEMORY = False

VAL_SUBSET = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# 🔥 Resume checkpoint (None nếu train từ đầu)
RESUME_CKPT = os.path.join(SAVE_DIR, "mimo_phase1_epoch_6.pth")
# =============================================================


def main():
    print("==============================================")
    print(">>> Phase 1: Drivable Area Segmentation")
    print(">>> Device:", DEVICE)
    print("==============================================")

    # ---------------- Dataset ----------------
    train_ds = BDD100kDriveDataset(
        root=DATA_ROOT,
        split="train",
        img_size=IMG_SIZE
    )
    val_ds = BDD100kDriveDataset(
        root=DATA_ROOT,
        split="val",
        img_size=IMG_SIZE
    )

    # 🔥 LIMIT validation size
    val_ds.images = val_ds.images[:VAL_SUBSET]
    val_ds.labels = val_ds.labels[:VAL_SUBSET]

    print(f">>> Train samples: {len(train_ds)}")
    print(f">>> Val samples:   {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # ---------------- Model ----------------
    model = MiMo(num_classes=1).to(DEVICE)

    start_epoch = 0
    best_val_loss = float("inf")

    # ---------------- Resume ----------------
    if RESUME_CKPT is not None and os.path.exists(RESUME_CKPT):
        print(f">>> Resuming from checkpoint: {RESUME_CKPT}")
        model.load_state_dict(torch.load(RESUME_CKPT, map_location=DEVICE))

        # cố gắng suy ra epoch từ tên file epoch_X
        try:
            for f in os.listdir(SAVE_DIR):
                if f.startswith("mimo_phase1_epoch_"):
                    e = int(f.split("_")[-1].split(".")[0])
                    start_epoch = max(start_epoch, e)
        except Exception:
            start_epoch = 0

        print(f">>> Resume from epoch {start_epoch}")

    model.train()

    # ---------------- Loss & Optim ----------------
    criterion = SegmentationLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ---------------- Training ----------------
    for epoch in range(start_epoch, EPOCHS):
        print(f"\n========== Epoch {epoch+1}/{EPOCHS} ==========")
        epoch_start = time.time()

        # -------- Train --------
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(DEVICE)
            targets = batch["drive_area"].to(DEVICE)

            outputs = model(images, mode="seg")
            loss = criterion(outputs["drive_area"], targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if step == 0:
                print(f"[Train] First batch loss: {loss.item():.4f}")

            if step % 100 == 0 and step > 0:
                print(f"[Train] Step {step}/{len(train_loader)}")

        train_loss /= len(train_loader)

        # -------- Validation --------
        print(">>> Start validation")
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i == 0:
                    print(">>> Validation batch 0 loaded")

                images = batch["image"].to(DEVICE)
                targets = batch["drive_area"].to(DEVICE)

                outputs = model(images, mode="seg")
                loss = criterion(outputs["drive_area"], targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(">>> End validation")

        # -------- Epoch summary --------
        elapsed = time.time() - epoch_start
        print(
            f"[Epoch {epoch+1}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {elapsed/60:.1f} min"
        )

        # -------- Save LAST --------
        last_path = os.path.join(SAVE_DIR, "mimo_phase1_last.pth")
        torch.save(model.state_dict(), last_path)

        # -------- Save per-epoch --------
        epoch_path = os.path.join(
            SAVE_DIR, f"mimo_phase1_epoch_{epoch+1}.pth"
        )
        torch.save(model.state_dict(), epoch_path)

        # -------- Save BEST --------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(SAVE_DIR, "mimo_phase1_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f">>> ✅ New BEST model saved (val_loss={val_loss:.4f})")

    print("\n✅ PHASE 1 TRAINING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
