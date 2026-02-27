import os
import time
import random
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

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
IMG_SIZE = 512
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0
PIN_MEMORY = False

VAL_SUBSET = 1000
SEED = 42
USE_AMP = True

# Hard-negative rebalancing (for reducing false positives)
ENABLE_HARD_NEG_REBALANCE = True
NEGATIVE_PIXEL_RATIO_THRESHOLD = 0.003
HARD_NEGATIVE_TARGET_RATIO = 0.35

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# None => train from scratch
RESUME_CKPT = os.path.join(SAVE_DIR, "mimo_phase1_last_full.pth")
# =============================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




def resolve_drivable_label_root(data_root: str):
    """
    Support BDD100K drivable label layouts:
    - labels/drivable/labels/{split}
    - labels/drivable/color_labels/{split}
    """
    candidates = [
        os.path.join(data_root, "labels", "drivable", "labels"),
        os.path.join(data_root, "labels", "drivable", "color_labels"),
    ]

    for cand in candidates:
        if os.path.isdir(cand):
            return cand

    raise FileNotFoundError(
        "Cannot find drivable label directory. Checked: " + ", ".join(candidates)
    )


def validate_data_layout(data_root: str):
    image_root = os.path.join(data_root, "images100k", "100k")
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    drivable_root = resolve_drivable_label_root(data_root)

    for split in ("train", "val"):
        img_split = os.path.join(image_root, split)
        lbl_split = os.path.join(drivable_root, split)
        if not os.path.isdir(img_split):
            raise FileNotFoundError(f"Image split not found: {img_split}")
        if not os.path.isdir(lbl_split):
            raise FileNotFoundError(f"Drivable label split not found: {lbl_split}")

    return image_root, drivable_root




def build_hard_negative_sampler(train_ds):
    """
    Build weighted sampler to oversample hard-negative samples
    (samples with near-empty drivable pixels).
    """
    if not ENABLE_HARD_NEG_REBALANCE:
        return None

    neg_flags = []

    for lbl_name in train_ds.labels:
        lbl_path = os.path.join(train_ds.lbl_dir, lbl_name)
        lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        if lbl is None:
            neg_flags.append(False)
            continue

        drive = ((lbl == 1) | (lbl == 2))
        drive_ratio = float(drive.mean())
        neg_flags.append(drive_ratio <= NEGATIVE_PIXEL_RATIO_THRESHOLD)

    n_total = len(neg_flags)
    n_neg = int(sum(neg_flags))
    n_pos = n_total - n_neg

    if n_total == 0 or n_neg == 0 or n_pos == 0:
        print(
            ">>> Hard-negative rebalance skipped "
            f"(total={n_total}, neg={n_neg}, pos={n_pos})"
        )
        return None

    target_ratio = min(max(HARD_NEGATIVE_TARGET_RATIO, 0.01), 0.99)

    pos_w = 1.0
    neg_w = (target_ratio * n_pos) / ((1.0 - target_ratio) * n_neg)

    sample_weights = [neg_w if is_neg else pos_w for is_neg in neg_flags]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=n_total,
        replacement=True,
    )

    est_neg_ratio = (neg_w * n_neg) / (neg_w * n_neg + pos_w * n_pos)
    print(
        ">>> Hard-negative rebalance enabled | "
        f"neg={n_neg}/{n_total} ({n_neg / n_total:.2%}) | "
        f"target_neg_ratio={target_ratio:.2f} | "
        f"estimated_sampled_neg_ratio={est_neg_ratio:.2f}"
    )

    return sampler


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
            "enable_hard_neg_rebalance": ENABLE_HARD_NEG_REBALANCE,
            "neg_pixel_ratio_threshold": NEGATIVE_PIXEL_RATIO_THRESHOLD,
            "hard_negative_target_ratio": HARD_NEGATIVE_TARGET_RATIO,
        },
    }


def save_checkpoint_pair(full_ckpt, model, full_path, weights_path):
    """
    Save both formats:
    - full checkpoint for training resume
    - weights-only state_dict for inference/visualization compatibility
    """
    torch.save(full_ckpt, full_path)
    torch.save(model.state_dict(), weights_path)


def try_resume(model, optimizer, scaler, ckpt_path):
    start_epoch = 0
    best_val_loss = float("inf")
    best_val_iou = 0.0

    if ckpt_path is None or not os.path.exists(ckpt_path):
        fallback = os.path.join(SAVE_DIR, "mimo_phase1_last.pth")
        if os.path.exists(fallback):
            ckpt_path = fallback
        else:
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
    image_root, drivable_root = validate_data_layout(DATA_ROOT)
    print(f">>> Image root:    {image_root}")
    print(f">>> Drivable root: {drivable_root}")

    train_ds = BDD100kDriveDataset(root=DATA_ROOT, split="train", img_size=IMG_SIZE)
    val_ds = BDD100kDriveDataset(root=DATA_ROOT, split="val", img_size=IMG_SIZE)

    # limit validation size
    if VAL_SUBSET is not None and VAL_SUBSET > 0:
        val_ds.images = val_ds.images[:VAL_SUBSET]
        val_ds.labels = val_ds.labels[:VAL_SUBSET]

    print(f">>> Train samples: {len(train_ds)}")
    print(f">>> Val samples:   {len(val_ds)}")

    train_sampler = build_hard_negative_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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

        train_iter = enumerate(train_loader)
        if tqdm is not None:
            train_iter = tqdm(
                train_iter,
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]",
                ncols=100,
                leave=True,
            )

        for step, batch in train_iter:
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

            if tqdm is not None:
                avg_train_loss = train_loss / (step + 1)
                train_iter.set_postfix(loss=f"{avg_train_loss:.4f}")

        train_loss /= max(len(train_loader), 1)

        # -------- Validation --------
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
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
        last_full_path = os.path.join(SAVE_DIR, "mimo_phase1_last_full.pth")
        last_weights_path = os.path.join(SAVE_DIR, "mimo_phase1_last.pth")
        save_checkpoint_pair(ckpt, model, last_full_path, last_weights_path)

        # -------- Save per-epoch --------
        epoch_full_path = os.path.join(SAVE_DIR, f"mimo_phase1_epoch_{epoch + 1}_full.pth")
        epoch_weights_path = os.path.join(SAVE_DIR, f"mimo_phase1_epoch_{epoch + 1}.pth")
        save_checkpoint_pair(ckpt, model, epoch_full_path, epoch_weights_path)

        # -------- Save BEST by val loss --------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt["best_val_loss"] = best_val_loss
            best_loss_full_path = os.path.join(SAVE_DIR, "mimo_phase1_best_loss_full.pth")
            best_loss_weights_path = os.path.join(SAVE_DIR, "mimo_phase1_best_loss.pth")
            save_checkpoint_pair(ckpt, model, best_loss_full_path, best_loss_weights_path)
            print(f">>> ✅ New BEST-LOSS model saved (val_loss={val_loss:.4f})")

        # -------- Save BEST by val IoU --------
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            ckpt["best_val_iou"] = best_val_iou
            best_iou_full_path = os.path.join(SAVE_DIR, "mimo_phase1_best_iou_full.pth")
            best_iou_weights_path = os.path.join(SAVE_DIR, "mimo_phase1_best_iou.pth")
            save_checkpoint_pair(ckpt, model, best_iou_full_path, best_iou_weights_path)
            print(f">>> ✅ New BEST-IOU model saved (val_iou={val_iou:.4f})")

    print("\n✅ PHASE 1 TRAINING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
