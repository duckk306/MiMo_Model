# LossModules.py (FIXED FOR PHASE-BASED TRAINING)

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.BboxUtils import decode_bbox, bbox_iou_ciou

# =========================== SEGMENTATION ===========================

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * inter + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    """
    Binary drivable-area loss (Phase 1, Phase 3)
    """
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        return (
            self.bce_weight * self.bce(pred, target)
            + self.dice_weight * self.dice(pred, target)
        )

# =========================== DETECTION ===========================

class FocalBCE(nn.Module):
    """
    Focal BCE for objectness / classification
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class DetectionClsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = FocalBCE()

    def forward(self, pred_cls, target_cls):
        return self.loss(pred_cls, target_cls)


class DetectionObjLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = FocalBCE(alpha=0.5)

    def forward(self, pred_obj, target_obj):
        return self.loss(pred_obj, target_obj)


class DetectionBoxLoss(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, pred_reg, target_reg, mask):
        mask = mask.bool()
        if mask.sum() == 0:
            return pred_reg.sum() * 0.0

        pred_boxes = decode_bbox(pred_reg, self.stride)
        target_boxes = decode_bbox(target_reg, self.stride)

        pred = pred_boxes.permute(0, 2, 3, 1)[mask.squeeze(1)]
        target = target_boxes.permute(0, 2, 3, 1)[mask.squeeze(1)]

        ciou = bbox_iou_ciou(pred, target)
        return (1.0 - ciou).mean()


class DetectionLoss(nn.Module):
    """
    Anchor-free detection loss
    """
    def __init__(self, strides=(8, 16, 32), box_weight=2.0, obj_weight=1.0):
        super().__init__()
        self.cls_loss = DetectionClsLoss()
        self.obj_loss = DetectionObjLoss()
        self.box_losses = nn.ModuleList(
            [DetectionBoxLoss(s) for s in strides]
        )
        self.box_weight = box_weight
        self.obj_weight = obj_weight

    def forward(self, preds, targets):
        total_cls = total_box = total_obj = 0.0

        for i, (p, t) in enumerate(zip(preds, targets)):
            total_cls += self.cls_loss(p["cls"], t["cls"])
            total_obj += self.obj_loss(p["obj"], t["obj"])

            mask = t["obj"] > 0
            total_box += self.box_losses[i](
                p["reg"], t["reg"], mask
            )

        return (
            total_cls
            + self.obj_weight * total_obj
            + self.box_weight * total_box
        )

# =========================== MULTI-TASK ===========================

class MultiTaskLoss(nn.Module):
    """
    Phase-aware multi-task loss
    """
    def __init__(self, lambda_seg=1.0):
        super().__init__()
        self.det_loss = DetectionLoss()
        self.seg_loss = SegmentationLoss()
        self.lambda_seg = lambda_seg

    def forward(self, outputs, targets):
        loss = 0.0

        if "detection" in outputs and targets.get("detection") is not None:
            loss += self.det_loss(
                outputs["detection"], targets["detection"]
            )

        if "drive_area" in outputs and targets.get("drive_area") is not None:
            loss += self.lambda_seg * self.seg_loss(
                outputs["drive_area"], targets["drive_area"]
            )

        return loss
