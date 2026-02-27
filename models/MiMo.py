# MiMo.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, act=True):
        super().__init__()
        if p is None:
            p = k // 2

        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# =========================== BACKBONE ===============================
class DWConvBlock(nn.Module):
    """
    Depthwise Separable Convolution Block
    """

    def __init__(self, in_ch, out_ch, stride=1, k=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=k,
                stride=stride,
                padding=k // 2,
                groups=in_ch,
                bias=False,
            ),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class RoadBackbone(nn.Module):
    """
    Shared Backbone
    Output:
        p2: 1/4 resolution (detail branch for segmentation)
        p4: 1/8 resolution
        p5: 1/16 resolution
        p6: 1/32 resolution
    """

    def __init__(self):
        super().__init__()
        self.stem = DWConvBlock(3, 32, stride=2, k=5)
        self.stage1 = DWConvBlock(32, 64, stride=2)
        self.stage2 = DWConvBlock(64, 128, stride=2)
        self.stage3 = DWConvBlock(128, 256, stride=2)
        self.stage4 = DWConvBlock(256, 512, stride=2)

    def forward(self, x):
        x = self.stem(x)  # 1/2
        p2 = self.stage1(x)  # 1/4
        p4 = self.stage2(p2)  # 1/8
        p5 = self.stage3(p4)  # 1/16
        p6 = self.stage4(p5)  # 1/32
        return p2, p4, p5, p6


# =========================== HEADS ==================================
class DetectHead(nn.Module):
    """
    Decoupled detect head:
    - cls tower
    - reg/obj tower
    """

    def __init__(self, in_ch, num_classes, width=128):
        super().__init__()

        self.cls_tower = nn.Sequential(
            ConvBNAct(in_ch, width, k=3),
            ConvBNAct(width, width, k=3),
        )
        self.reg_tower = nn.Sequential(
            ConvBNAct(in_ch, width, k=3),
            ConvBNAct(width, width, k=3),
        )

        self.cls = nn.Conv2d(width, num_classes, 1)
        self.obj = nn.Conv2d(width, 1, 1)
        self.reg = nn.Conv2d(width, 4, 1)

    def forward(self, x):
        cls_feat = self.cls_tower(x)
        reg_feat = self.reg_tower(x)

        return {
            "obj": self.obj(reg_feat),
            "cls": self.cls(cls_feat),
            "reg": self.reg(reg_feat),
        }


class DriveAreaHead(nn.Module):
    """
    Segmentation head with detail skip (p2) + semantic feature (p4).
    """

    def __init__(self, in_ch=128, detail_ch=64, out_ch=1):
        super().__init__()

        self.semantic_proj = ConvBNAct(in_ch, 128, k=3)
        self.detail_proj = ConvBNAct(detail_ch, 64, k=3)

        self.fuse = nn.Sequential(
            ConvBNAct(128 + 64, 128, k=3),
            ConvBNAct(128, 64, k=3),
            nn.Conv2d(64, out_ch, 1),
        )

    def forward(self, p4, p2, out_size):
        semantic = self.semantic_proj(p4)
        semantic = F.interpolate(semantic, size=p2.shape[2:], mode="bilinear", align_corners=False)

        detail = self.detail_proj(p2)
        fused = torch.cat([semantic, detail], dim=1)
        logits = self.fuse(fused)

        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


# =========================== NECK ===================================
class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce_p6 = nn.Conv2d(512, 256, 1)
        self.reduce_p5 = nn.Conv2d(256, 128, 1)

        # refinement after each fusion to improve feature quality
        self.refine_p5 = ConvBNAct(256, 256, k=3)
        self.refine_p4 = ConvBNAct(128, 128, k=3)

    def forward(self, p4, p5, p6):
        p6 = self.reduce_p6(p6)

        p6_up = F.interpolate(p6, scale_factor=2, mode="nearest")
        p5 = self.refine_p5(p5 + p6_up)

        p5_reduced = self.reduce_p5(p5)
        p5_up = F.interpolate(p5_reduced, scale_factor=2, mode="nearest")
        p4 = self.refine_p4(p4 + p5_up)

        return p4, p5_reduced, p6


# =========================== MODEL ==================================
class MiMo(nn.Module):
    """
    Multi-task model:
    - Detection
    - Drive-able area segmentation

    Args:
        num_classes: detection class count
        seg_classes: segmentation output channels.
                     Default=1 keeps backward compatibility for phase-1 binary training.
    """

    def __init__(self, num_classes, seg_classes=1):
        super().__init__()
        self.backbone = RoadBackbone()
        self.neck = Neck()

        self.det_s = DetectHead(128, num_classes)
        self.det_m = DetectHead(128, num_classes)
        self.det_l = DetectHead(256, num_classes)

        self.seg_head = DriveAreaHead(in_ch=128, detail_ch=64, out_ch=seg_classes)

    def forward(self, x, mode="multi"):
        H, W = x.shape[2:]

        p2, p4, p5, p6 = self.backbone(x)
        p4, p5, p6 = self.neck(p4, p5, p6)

        outputs = {}

        if mode in ("det", "multi"):
            outputs["detection"] = [
                self.det_s(p4),
                self.det_m(p5),
                self.det_l(p6),
            ]

        if mode in ("seg", "multi"):
            outputs["drive_area"] = self.seg_head(p4, p2, (H, W))

        return outputs
