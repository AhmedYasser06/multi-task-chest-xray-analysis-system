"""Dice loss for the segmentation head, translated from utils/Segmenter.py."""

import torch
import torch.nn as nn

from .. import config


def dice_coeff(y_pred, y_true, smooth=1.0):
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    y_true = y_true.reshape(y_true.shape[0], -1)
    intersection = (y_pred * y_true).sum(dim=1)
    return (2.0 * intersection + smooth) / (
        y_pred.square().sum(dim=1) + y_true.square().sum(dim=1) + smooth
    )


class DiceLoss(nn.Module):
    def __init__(self, ignore_value=config.IGNORE_VALUE):
        super().__init__()
        self.ignore_value = ignore_value

    def forward(self, y_pred, y_true):
        """Returns a (B,) per-sample loss. Samples whose y_true is entirely
        `ignore_value` (i.e. no segmentation label in this batch item)
        contribute 0."""
        b = y_true.shape[0]
        valid = ~torch.all(y_true.view(b, -1) == self.ignore_value, dim=1)

        loss = 1.0 - dice_coeff(y_pred, torch.clamp(y_true, min=0.0))
        return loss * valid.float()


class BCEDiceLoss(nn.Module):
    """BCE + Dice combo loss.

    Plain Dice loss on a heavily-imbalanced dataset (most chest x-rays have
    an entirely empty pneumothorax mask) has a degenerate optimum: predicting
    all-zero already gives dice~1 on ~80% of samples, so gradients vanish and
    the model gets stuck outputting a near-blank mask after epoch 1. Mixing
    in per-pixel BCE keeps a real gradient flowing on every pixel regardless
    of how saturated the Dice term is - the standard fix for this failure
    mode on SIIM-ACR-style segmentation tasks.

    `pos_weight` up-weights the (rare) positive pixels inside the BCE term to
    further counteract the class imbalance.
    """

    def __init__(self, ignore_value=config.IGNORE_VALUE, bce_weight=0.5,
                 dice_weight=0.5, pos_weight=4.0):
        super().__init__()
        self.ignore_value = ignore_value
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = torch.tensor(pos_weight) if pos_weight is not None else None
        self.bce = nn.BCELoss(reduction="none")

    def forward(self, y_pred, y_true):
        b = y_true.shape[0]
        valid = ~torch.all(y_true.view(b, -1) == self.ignore_value, dim=1)
        y_true_clamped = torch.clamp(y_true, min=0.0)

        dice = 1.0 - dice_coeff(y_pred, y_true_clamped)

        # nn.BCELoss (probabilities in, not logits) is explicitly disallowed
        # under autocast/mixed precision - PyTorch raises
        # "binary_cross_entropy and BCELoss are unsafe to autocast" because
        # it's numerically unstable in fp16. The segmenter head already
        # applies sigmoid internally, so y_pred here is a probability, not a
        # logit - we can't just swap in BCEWithLogitsLoss without changing
        # the model. Instead, force this specific computation to run in fp32
        # outside of autocast, which is exactly what BCEWithLogitsLoss does
        # internally anyway.
        with torch.autocast(device_type=y_pred.device.type, enabled=False):
            y_pred_f32 = y_pred.float()
            y_true_f32 = y_true_clamped.float()
            bce_map = self.bce(y_pred_f32, y_true_f32)
            if self.pos_weight is not None:
                pw = self.pos_weight.to(y_pred.device)
                weight_map = 1.0 + (pw - 1.0) * y_true_f32
                bce_map = bce_map * weight_map
            bce = bce_map.reshape(b, -1).mean(dim=1)

        loss = self.bce_weight * bce + self.dice_weight * dice
        return loss * valid.float()
