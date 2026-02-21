# sanity_test.py (PHASE 1 – SEG ONLY)

import torch

from models.MiMo import MiMo
from losses.LossModules import SegmentationLoss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1️⃣ Init model (Phase 1: segmentation only)
    model = MiMo(num_classes=1).to(device)
    model.train()

    # 2️⃣ Fake input
    x = torch.randn(2, 3, 640, 640, device=device)

    # 3️⃣ Forward (SEG MODE)
    outputs = model(x, mode="seg")

    assert "drive_area" in outputs
    assert outputs["drive_area"].shape == (2, 1, 640, 640)

    print("[OK] Forward (seg)")

    # 4️⃣ Fake target
    targets = torch.zeros_like(outputs["drive_area"])

    # 5️⃣ Loss (SEG ONLY)
    criterion = SegmentationLoss()
    loss = criterion(outputs["drive_area"], targets)

    print("[OK] Loss computed:", loss.item())

    # 6️⃣ Backward
    loss.backward()

    print("[OK] Backward pass")

    # 7️⃣ Gradient check
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN/Inf grad in {name}"

    print("[OK] Gradients finite")

    print("\n🔥 PHASE 1 SANITY CHECK PASSED 🔥")


if __name__ == "__main__":
    main()
