"""
Tiny-YOLOv2 style loss, translated from utils/Detector.py (yolo_loss).

y_true / y_pred shape: (B, S, S, n_anchors, 5 + n_classes)
    [..., 0:2] = box center (tx, ty) in grid-cell units (already offset by
                 cell index for y_true; raw for y_pred, sigmoid applied here)
    [..., 2:4] = box width/height (in anchor-units, exp() applied to y_pred)
    [..., 4]   = objectness (1 if a box is responsible for this cell/anchor)
    [..., 5:]  = one-hot class scores

If an entire sample's y_true is filled with config.IGNORE_VALUE (-1), that
sample contributes zero loss (used for the joint MTL training loop where a
batch item might come from a dataset without detection labels).
"""

import torch
import torch.nn as nn

from .. import config


class YoloLoss(nn.Module):
    def __init__(self, anchors=config.ANCHORS, object_coord_scale=5.0,
                 object_conf_scale=1.0, noobject_conf_scale=1.0,
                 object_class_scale=1.0, ignore_value=config.IGNORE_VALUE):
        super().__init__()
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32))
        self.object_coord_scale = object_coord_scale
        self.object_conf_scale = object_conf_scale
        self.noobject_conf_scale = noobject_conf_scale
        self.object_class_scale = object_class_scale
        self.ignore_value = ignore_value

    def forward(self, y_pred, y_true):
        """Returns a (B,) tensor: per-sample loss (0 for ignored samples)."""
        device = y_pred.device
        b, n_cells, _, n_anchors, _ = y_pred.shape
        anchors = self.anchors.to(device)

        valid = ~torch.all(y_true.view(b, -1) == self.ignore_value, dim=1)  # (B,)

        # ---- decode predictions ----
        pred_xy = torch.sigmoid(y_pred[..., 0:2])
        cell_inds = torch.arange(n_cells, device=device, dtype=torch.float32)
        grid_x = cell_inds.view(1, 1, -1, 1)
        grid_y = cell_inds.view(1, -1, 1, 1)
        pred_x = pred_xy[..., 0] + grid_x
        pred_y = pred_xy[..., 1] + grid_y
        pred_xy = torch.stack([pred_x, pred_y], dim=-1)

        # clamp(max=10) before exp(): under fp16 autocast, exp() of a large
        # pre-activation silently overflows to inf (confirmed: exp(20) in
        # fp16 -> inf), which then poisons the loss/gradients with NaN with
        # no error raised. exp(10)=~22026, already far larger than any real
        # anchor-relative box size, so this only clips runaway values, not
        # normal training dynamics.
        pred_wh = anchors.view(1, 1, 1, n_anchors, 2) * torch.exp(
            torch.clamp(y_pred[..., 2:4], max=10.0))
        pred_min = pred_xy - pred_wh / 2
        pred_max = pred_xy + pred_wh / 2

        pred_obj = torch.sigmoid(y_pred[..., 4])
        pred_cls = torch.softmax(y_pred[..., 5:], dim=-1)

        # ---- ground truth ----
        true_xy = y_true[..., 0:2]
        true_wh = y_true[..., 2:4]
        true_cls = y_true[..., 5:]
        true_min = true_xy - true_wh / 2
        true_max = true_xy + true_wh / 2

        # ---- IoU between predicted and true boxes (for objectness target) ----
        inter_min = torch.maximum(pred_min, true_min)
        inter_max = torch.minimum(pred_max, true_max)
        inter_wh = torch.clamp(inter_max - inter_min, min=0.0)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]

        true_area = true_wh[..., 0] * true_wh[..., 1]
        pred_area = pred_wh[..., 0] * pred_wh[..., 1]
        union_area = pred_area + true_area - inter_area + 1e-9
        iou = inter_area / union_area

        responsible = y_true[..., 4]                      # (B,S,S,A) in {0,1}

        xy_loss = (torch.square(true_xy - pred_xy) * responsible.unsqueeze(-1)).sum(dim=[1, 2, 3, 4])
        wh_loss = (torch.square(torch.sqrt(torch.clamp(true_wh, min=0)) -
                                 torch.sqrt(torch.clamp(pred_wh, min=1e-9)))
                   * responsible.unsqueeze(-1)).sum(dim=[1, 2, 3, 4])

        obj_loss = (torch.square(iou - pred_obj) * responsible).sum(dim=[1, 2, 3])

        best_iou = iou.amax(dim=-1, keepdim=True)                      # (B,S,S,1)
        no_obj_mask = (best_iou < 0.6).float() * (1 - responsible)     # (B,S,S,A)
        no_obj_loss = (torch.square(0 - pred_obj) * no_obj_mask).sum(dim=[1, 2, 3])

        cls_loss = (torch.square(true_cls - pred_cls) * responsible.unsqueeze(-1)).sum(dim=[1, 2, 3, 4])

        loss = (self.object_coord_scale * (xy_loss + wh_loss)
                + self.object_conf_scale * obj_loss
                + self.noobject_conf_scale * no_obj_loss
                + self.object_class_scale * cls_loss)

        loss = loss * valid.float()
        return loss   # per-sample; caller decides how to reduce (mean over valid, etc.)
