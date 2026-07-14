"""Simple metrics used during training / evaluation notebooks."""

import torch


@torch.no_grad()
def classification_accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def dice_score(pred_mask, true_mask, threshold=0.5, eps=1e-6):
    """pred_mask, true_mask: (B,1,H,W) in [0,1]."""
    pred_bin = (pred_mask > threshold).float()
    true_bin = (true_mask > threshold).float()
    intersection = (pred_bin * true_bin).sum(dim=[1, 2, 3])
    union = pred_bin.sum(dim=[1, 2, 3]) + true_bin.sum(dim=[1, 2, 3])
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


@torch.no_grad()
def dice_score_positive_sum(pred_mask, true_mask, threshold=0.5, eps=1e-6):
    """Per-sample dice, but only summed over samples whose ground-truth mask
    actually contains a positive pixel.

    Because ~80% of pneumothorax masks are entirely empty, `dice_score()`
    above is dominated by "predict nothing" samples which trivially score
    ~1.0, hiding whether the model can actually segment the disease region.
    This returns (sum_of_dice_over_positive_samples, n_positive_samples) for
    a batch so callers can accumulate a true epoch-level average over only
    the samples that matter for judging segmentation quality.
    """
    pred_bin = (pred_mask > threshold).float()
    true_bin = (true_mask > threshold).float()
    has_positive = true_bin.sum(dim=[1, 2, 3]) > 0

    intersection = (pred_bin * true_bin).sum(dim=[1, 2, 3])
    union = pred_bin.sum(dim=[1, 2, 3]) + true_bin.sum(dim=[1, 2, 3])
    dice = (2 * intersection + eps) / (union + eps)

    dice_pos = dice[has_positive]
    return dice_pos.sum().item(), int(has_positive.sum().item())


@torch.no_grad()
def iou_score(pred_mask, true_mask, threshold=0.5, eps=1e-6):
    pred_bin = (pred_mask > threshold).float()
    true_bin = (true_mask > threshold).float()
    intersection = (pred_bin * true_bin).sum(dim=[1, 2, 3])
    union = ((pred_bin + true_bin) > 0).float().sum(dim=[1, 2, 3])
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def box_iou_xyxy(box_a, box_b):
    """box_a, box_b: [x1,y1,x2,y2]."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter + 1e-9
    return inter / union
