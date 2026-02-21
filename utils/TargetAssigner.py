# utils/TargetAssigner.py

import torch
import math


class TargetAssigner:
    """
    Anchor-free, center-based assigner
    """
    def __init__(self, num_classes, img_size=640, strides=(8, 16, 32)):
        self.num_classes = num_classes
        self.img_size = img_size
        self.strides = strides

    def build_targets(self, gt_boxes, gt_labels, device):
        B = len(gt_boxes)
        targets = []

        for stride in self.strides:
            H = self.img_size // stride
            W = self.img_size // stride
            targets.append({
                "obj": torch.zeros((B, 1, H, W), device=device),
                "cls": torch.zeros((B, self.num_classes, H, W), device=device),
                "reg": torch.zeros((B, 4, H, W), device=device),
            })

        for b in range(B):
            for box, label in zip(gt_boxes[b], gt_labels[b]):
                cx, cy, w, h = box.tolist()

                scale = self.choose_scale(w, h)
                stride = self.strides[scale]

                gi = int(cx // stride)
                gj = int(cy // stride)

                H = self.img_size // stride
                W = self.img_size // stride
                if gi < 0 or gj < 0 or gi >= W or gj >= H:
                    continue

                # prevent overwrite
                if targets[scale]["obj"][b, 0, gj, gi] == 1:
                    continue

                tx = (cx / stride) - gi
                ty = (cy / stride) - gj
                tw = math.log(max(w / stride, 1e-4))
                th = math.log(max(h / stride, 1e-4))

                targets[scale]["obj"][b, 0, gj, gi] = 1.0
                targets[scale]["cls"][b, label, gj, gi] = 1.0
                targets[scale]["reg"][b, :, gj, gi] = torch.tensor(
                    [tx, ty, tw, th], device=device
                )

        return targets

    def choose_scale(self, w, h):
        size = max(w, h)
        if size <= 48:
            return 0
        elif size <= 96:
            return 1
        else:
            return 2
