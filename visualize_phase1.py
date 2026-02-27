import os
import cv2
import torch
import numpy as np

from models.MiMo import MiMo


# ========================= CONFIG =============================
IMAGE_PATH = "traffic.jpg"  # đổi sang ảnh test của bạn
CHECKPOINT = ""  # để trống -> tự dò theo thứ tự ưu tiên
OUT_PATH = "vis_phase1_overlay.jpg"
SAVE_OUTPUT = False  # True nếu bạn vẫn muốn lưu file ảnh
WINDOW_NAME = "MiMo Phase1 Visualization"
IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESH = 0.5
# =============================================================


# OpenCV runtime stability
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_checkpoint_path(checkpoint_path):
    """
    Phù hợp với naming trong train_phase1.py:
    - mimo_phase1_best_loss.pth / _full.pth
    - mimo_phase1_best_iou.pth / _full.pth
    - mimo_phase1_last.pth / _full.pth
    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        return checkpoint_path

    candidates = [
        "checkpoints/mimo_phase1_best_loss.pth",
        "checkpoints/mimo_phase1_best_loss_full.pth",
        "checkpoints/mimo_phase1_best_iou.pth",
        "checkpoints/mimo_phase1_best_iou_full.pth",
        "checkpoints/mimo_phase1_last.pth",
        "checkpoints/mimo_phase1_last_full.pth",
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        "No checkpoint found. Checked: " + ", ".join(candidates)
    )


def load_checkpoint_compatible(checkpoint_path, map_location):
    """
    Hỗ trợ cả 2 format checkpoint từ train_phase1.py:
    1) weights-only state_dict
    2) full checkpoint dict: {'model_state': ..., 'config': ...}

    Returns:
        state_dict, meta(dict)
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        print(">>> Detected full checkpoint format. Using key: model_state")
        meta = {"config": ckpt.get("config", {})}
        return ckpt["model_state"], meta

    if isinstance(ckpt, dict):
        # weights-only dict
        sample_keys = list(ckpt.keys())[:3]
        print(f">>> Detected weights state_dict. Sample keys: {sample_keys}")
        return ckpt, {"config": {}}

    raise RuntimeError(
        "Unsupported checkpoint format. Expected state_dict or dict with 'model_state'."
    )


def infer_seg_classes_from_state_dict(state_dict):
    # theo kiến trúc hiện tại: seg_head.fuse.2.weight -> [out_ch, 64, 1, 1]
    key = "seg_head.fuse.2.weight"
    if key in state_dict:
        return int(state_dict[key].shape[0])
    return 1


def preprocess_image(image_bgr, img_size):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (img_size, img_size))
    x = resized.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    return x


def postprocess_mask(logits, out_h, out_w, thresh):
    # binary seg (phase 1): [B,1,H,W]
    if logits.shape[1] == 1:
        probs = torch.sigmoid(logits)
        mask = (probs > thresh).float()[0, 0].cpu().numpy()
    else:
        # multi-class fallback: lấy class khác background
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy()
        mask = (pred > 0).astype(np.float32)

    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return mask


def overlay_mask(image_bgr, mask, alpha=0.45):
    color = np.zeros_like(image_bgr)
    color[:, :, 1] = mask  # green mask
    return cv2.addWeighted(image_bgr, 1.0, color, alpha, 0)


def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    checkpoint_path = resolve_checkpoint_path(CHECKPOINT)

    print("==============================================")
    print(">>> Visualize Phase 1 - Drivable Area")
    print(">>> Device:", DEVICE)
    print(">>> Checkpoint:", checkpoint_path)
    print("==============================================")

    state_dict, meta = load_checkpoint_compatible(checkpoint_path, map_location=DEVICE)

    img_size = int(meta.get("config", {}).get("img_size", IMG_SIZE))
    seg_classes = infer_seg_classes_from_state_dict(state_dict)

    # Build model (auto infer seg_classes từ checkpoint)
    model = MiMo(num_classes=1, seg_classes=seg_classes).to(DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Read image
    image_bgr = cv2.imread(IMAGE_PATH)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {IMAGE_PATH}")

    h, w = image_bgr.shape[:2]
    x = preprocess_image(image_bgr, img_size).to(DEVICE)

    with torch.no_grad():
        outputs = model(x, mode="seg")
        logits = outputs["drive_area"]

    mask = postprocess_mask(logits, h, w, THRESH)
    overlay = overlay_mask(image_bgr, mask)

    if SAVE_OUTPUT:
        cv2.imwrite(OUT_PATH, overlay)
        print(f">>> Saved overlay: {OUT_PATH}")

    print(">>> Press any key to close window...")
    cv2.imshow(WINDOW_NAME, overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
