#MiMo.py

import torch.nn as nn
import torch.nn.functional as F

#===========================BACKBONE===============================================
class DWConvBlock(nn.Module):
    """
    Depthwise Separable Convolution Block
    """
    def __init__(self, in_ch, out_ch, stride=1, k=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch, in_ch, kernel_size=k,
                stride=stride, padding=k//2,
                groups=in_ch, bias=False
            ),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),

            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class RoadBackbone(nn.Module):
    """
    Shared Backbone
    Output:
        P4: 1/8 resolution
        P5: 1/16 resolution
        P6: 1/32 resolution
    """
    def __init__(self):
        super().__init__()
        self.stem = DWConvBlock(3, 32, stride=2, k=5)
        self.stage1 = DWConvBlock(32, 64, stride=2)
        self.stage2 = DWConvBlock(64, 128, stride=2)
        self.stage3 = DWConvBlock(128, 256, stride=2)
        self.stage4 = DWConvBlock(256, 512, stride=2)

    def forward(self, x):
        x = self.stem(x)        # 1/2
        x = self.stage1(x)     # 1/4
        p4 = self.stage2(x)    # 1/8
        p5 = self.stage3(p4)   # 1/16
        p6 = self.stage4(p5)   # 1/32
        return p4, p5, p6

#===========================HEADS===============================================
class DetectHead(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()

        self.obj = nn.Conv2d(in_ch, 1, 1)
        self.cls = nn.Conv2d(in_ch, num_classes, 1)
        self.reg = nn.Conv2d(in_ch, 4, 1)

    def forward(self, x):
        return {
            "obj": self.obj(x),
            "cls": self.cls(x),
            "reg": self.reg(x),
        }

class DriveAreaHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, 1)   # 🔥 CHỐT: 1 channel
        )

    def forward(self, x, out_size):
        x = self.decoder(x)
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)


#===========================NECK===============================================
class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce_p6 = nn.Conv2d(512, 256, 1)
        self.reduce_p5 = nn.Conv2d(256, 128, 1)

    def forward(self, p4, p5, p6):
        p6 = self.reduce_p6(p6)
        p6_up = F.interpolate(p6, scale_factor=2, mode="nearest")
        p5 = p5 + p6_up

        p5 = self.reduce_p5(p5)
        p5_up = F.interpolate(p5, scale_factor=2, mode="nearest")
        p4 = p4 + p5_up

        return p4, p5, p6


#===========================MODEL===============================================
class MiMo(nn.Module):
    """
    Multi-task model:
    - Detection
    - Drive-able area segmentation
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = RoadBackbone()
        self.neck = Neck()

        self.det_s = DetectHead(128, num_classes)
        self.det_m = DetectHead(128, num_classes)
        self.det_l = DetectHead(256, num_classes)

        self.seg_head = DriveAreaHead(128)

    def forward(self, x, mode="multi"):
        H, W = x.shape[2:]

        p4, p5, p6 = self.backbone(x)
        p4, p5, p6 = self.neck(p4, p5, p6)

        outputs = {}

        if mode in ("det", "multi"):
            outputs["detection"] = [
                self.det_s(p4),
                self.det_m(p5),
                self.det_l(p6),
            ]

        if mode in ("seg", "multi"):
            outputs["drive_area"] = self.seg_head(p4, (H, W))

        return outputs
