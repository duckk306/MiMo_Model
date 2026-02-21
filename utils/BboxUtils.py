# utils/BboxUtils.py (FIXED, NUMERICALLY SAFE)

import torch

def decode_bbox(pred_reg, stride):
    """
    pred_reg: (B,4,H,W) -> (cx,cy,w,h)
    """
    B, _, H, W = pred_reg.shape
    device = pred_reg.device

    tx, ty, tw, th = pred_reg.unbind(1)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )

    cx = (grid_x + tx) * stride
    cy = (grid_y + ty) * stride

    tw = tw.clamp(min=-4.0, max=4.0)
    th = th.clamp(min=-4.0, max=4.0)

    w = torch.exp(tw) * stride
    h = torch.exp(th) * stride

    return torch.stack([cx, cy, w, h], dim=1)


def bbox_iou_ciou(pred, target, eps=1e-7):
    px, py, pw, ph = pred.T
    gx, gy, gw, gh = target.T

    p_x1, p_y1 = px - pw / 2, py - ph / 2
    p_x2, p_y2 = px + pw / 2, py + ph / 2
    g_x1, g_y1 = gx - gw / 2, gy - gh / 2
    g_x2, g_y2 = gx + gw / 2, gy + gh / 2

    inter = (
        (torch.min(p_x2, g_x2) - torch.max(p_x1, g_x1)).clamp(0)
        * (torch.min(p_y2, g_y2) - torch.max(p_y1, g_y1)).clamp(0)
    )

    union = pw * ph + gw * gh - inter + eps
    iou = inter / union

    center_dist = (px - gx) ** 2 + (py - gy) ** 2

    enc_x1 = torch.min(p_x1, g_x1)
    enc_y1 = torch.min(p_y1, g_y1)
    enc_x2 = torch.max(p_x2, g_x2)
    enc_y2 = torch.max(p_y2, g_y2)
    enc_diag = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    v = (4 / (torch.pi ** 2)) * (
        torch.atan(gw / gh) - torch.atan(pw / ph)
    ) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    return iou - center_dist / enc_diag - alpha * v
