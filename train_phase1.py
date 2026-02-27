import os
import time
import random
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.MiMo import MiMo
from datasets.BDD100kDriveDataset import BDD100kDriveDataset
from losses.LossModules import SegmentationLoss


# ======================= CRITICAL FIXES =======================
# OpenCV + Windows + PyTorch fix
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
# =============================================================


# ========================= CONFIG =============================
DATA_ROOT = "data/bdd100k"
IMG_SIZE = 448
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 2
PIN_MEMORY = True

VAL_SUBSET = 500
SEED = 42
USE_AMP = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# None => train from scratch
RESUME_CKPT = os.path.join(SAVE_DIR, "mimo_phase1_last.pth")
# =============================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_seg_metrics(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    """
    Binary segmentation metrics from logits:
    - IoU
    - Dice/F1
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    inter = (preds * targets).sum(dim=(1, 2, 3))
    pred_sum = preds.sum(dim=(1, 2, 3))
    tgt_sum = targets.sum(dim=(1, 2, 3))
    union = pred_sum + tgt_sum - inter

    iou = ((inter + eps) / (union + eps)).mean().item()
    dice = ((2.0 * inter + eps) / (pred_sum + tgt_sum + eps)).mean().item()
    return iou, dice


def build_checkpoint(
    epoch,
    model,
    optimizer,
    scaler,
    best_val_loss,
    best_val_iou,
):
    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "best_val_iou": best_val_iou,
        "config": {
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "use_amp": USE_AMP,
            "seed": SEED,
        },
    }


def try_resume(model, optimizer, scaler, ckpt_path):
    start_epoch = 0
    best_val_loss = float("inf")
    best_val_iou = 0.0

    if ckpt_path is None or not os.path.exists(ckpt_path):
        return start_epoch, best_val_loss, best_val_iou

    print(f">>> Resuming from checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # backward compatibility: old checkpoints may store only state_dict
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scaler is not None and ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])

        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        best_val_iou = float(ckpt.get("best_val_iou", 0.0))
    else:
        model.load_state_dict(ckpt)

    print(f">>> Resume from epoch: {start_epoch}")
    print(f">>> Best val loss so far: {best_val_loss:.4f}")
    print(f">>> Best val IoU so far:  {best_val_iou:.4f}")
    return start_epoch, best_val_loss, best_val_iou


def main():
    set_seed(SEED)

    print("==============================================")
    print(">>> Phase 1: Drivable Area Segmentation")
    print(">>> Device:", DEVICE)
    print(">>> AMP:", USE_AMP and DEVICE == "cuda")
    print("==============================================")

    # ---------------- Dataset ----------------
    train_ds = BDD100kDriveDataset(root=DATA_ROOT, split="train", img_size=IMG_SIZE)
    val_ds = BDD100kDriveDataset(root=DATA_ROOT, split="val", img_size=IMG_SIZE)

    # limit validation size
    if VAL_SUBSET is not None and VAL_SUBSET > 0:
        val_ds.images = val_ds.images[:VAL_SUBSET]
        val_ds.labels = val_ds.labels[:VAL_SUBSET]

    print(f">>> Train samples: {len(train_ds)}")
    print(f">>> Val samples:   {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    # ---------------- Model ----------------
    model = MiMo(num_classes=1, seg_classes=1).to(DEVICE)

    # ---------------- Loss & Optim ----------------
    criterion = SegmentationLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    amp_enabled = USE_AMP and DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None

    # ---------------- Resume ----------------
    start_epoch, best_val_loss, best_val_iou = try_resume(model, optimizer, scaler, RESUME_CKPT)

    # ---------------- Training ----------------
    for epoch in range(start_epoch, EPOCHS):
        print(f"\n========== Epoch {epoch + 1}/{EPOCHS} ==========")
        epoch_start = time.time()

        # -------- Train --------
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(DEVICE, non_blocking=True)
            targets = batch["drive_area"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if amp_enabled:
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    outputs = model(images, mode="seg")
                    loss = criterion(outputs["drive_area"], targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images, mode="seg")
                loss = criterion(outputs["drive_area"], targets)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

            if step == 0:
                print(f"[Train] First batch loss: {loss.item():.4f}")
            if step % 100 == 0 and step > 0:
                print(f"[Train] Step {step}/{len(train_loader)}")

        train_loss /= max(len(train_loader), 1)

        # -------- Validation --------
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i == 0:
                    print(">>> Validation started")

                images = batch["image"].to(DEVICE, non_blocking=True)
                targets = batch["drive_area"].to(DEVICE, non_blocking=True)

                if amp_enabled:
                    with torch.amp.autocast(device_type="cuda", enabled=True):
                        outputs = model(images, mode="seg")
                        logits = outputs["drive_area"]
                        loss = criterion(logits, targets)
                else:
                    outputs = model(images, mode="seg")
                    logits = outputs["drive_area"]
                    loss = criterion(logits, targets)

                iou, dice = compute_seg_metrics(logits, targets)
                val_loss += loss.item()
                val_iou += iou
                val_dice += dice

        num_val_batches = max(len(val_loader), 1)
        val_loss /= num_val_batches
        val_iou /= num_val_batches
        val_dice /= num_val_batches

        # -------- Epoch summary --------
        elapsed = time.time() - epoch_start
        print(
            f"[Epoch {epoch + 1}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Time: {elapsed / 60:.1f} min"
        )

        ckpt = build_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            best_val_loss=best_val_loss,
            best_val_iou=best_val_iou,
        )

        # -------- Save LAST --------
        last_path = os.path.join(SAVE_DIR, "mimo_phase1_last.pth")
        torch.save(ckpt, last_path)

        # -------- Save per-epoch --------
        epoch_path = os.path.join(SAVE_DIR, f"mimo_phase1_epoch_{epoch + 1}.pth")
        torch.save(ckpt, epoch_path)

        # -------- Save BEST by val loss --------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt["best_val_loss"] = best_val_loss
            best_loss_path = os.path.join(SAVE_DIR, "mimo_phase1_best_loss.pth")
            torch.save(ckpt, best_loss_path)
            print(f">>> ✅ New BEST-LOSS model saved (val_loss={val_loss:.4f})")

        # -------- Save BEST by val IoU --------
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            ckpt["best_val_iou"] = best_val_iou
            best_iou_path = os.path.join(SAVE_DIR, "mimo_phase1_best_iou.pth")
            torch.save(ckpt, best_iou_path)
            print(f">>> ✅ New BEST-IOU model saved (val_iou={val_iou:.4f})")

    print("\n✅ PHASE 1 TRAINING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
