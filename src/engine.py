"""
Training / evaluation loops. Kept deliberately simple and explicit so the
notebooks can just import these functions instead of re-writing the loop in
every notebook.
"""

import torch
from tqdm.auto import tqdm

from . import config
from .losses import YoloLoss, DiceLoss, BCEDiceLoss
from .utils.metrics import classification_accuracy, dice_score, dice_score_positive_sum


# ---------------------------------------------------------------------------
# Single-head training (used by 01/02/03 notebooks)
# ---------------------------------------------------------------------------
def train_one_epoch_segmenter(model, loader, optimizer, device=config.DEVICE, scaler=None,
                               pos_weight="auto"):
    """Uses BCE+Dice instead of plain Dice: on SIIM-ACR ~80% of masks are
    entirely empty, so plain Dice loss collapses to a "predict nothing"
    optimum within ~1 epoch and gradients vanish. BCE keeps a real per-pixel
    gradient flowing; `pos_weight` up-weights positive (pneumothorax) pixels
    inside the BCE term to counteract the imbalance further.

    Reports both the overall dice (`dice`, dominated by trivially-correct
    empty-mask samples) and `pos_dice` (dice averaged only over samples that
    actually contain pneumothorax) - watch `pos_dice`/`val_pos_dice` as the
    real signal of whether the model is learning to segment."""
    model.train()
    loss_fn = BCEDiceLoss(pos_weight=pos_weight)
    running_loss, running_dice = 0.0, 0.0
    pos_dice_sum, pos_dice_n = 0.0, 0

    for img, mask, _class_label in tqdm(loader, desc="train[seg]", leave=False):
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                             enabled=(scaler is not None)):
            out = model(img)
            pred_mask = out["seg"] if isinstance(out, dict) else out
            loss = loss_fn(pred_mask, mask).mean()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        running_dice += dice_score(pred_mask.detach(), mask)
        d_sum, d_n = dice_score_positive_sum(pred_mask.detach(), mask)
        pos_dice_sum += d_sum
        pos_dice_n += d_n

    n = len(loader)
    return {
        "loss": running_loss / n,
        "dice": running_dice / n,
        "pos_dice": pos_dice_sum / max(pos_dice_n, 1),
    }


@torch.no_grad()
def evaluate_segmenter(model, loader, device=config.DEVICE, pos_weight="auto"):
    model.eval()
    loss_fn = BCEDiceLoss(pos_weight=pos_weight)
    running_loss, running_dice = 0.0, 0.0
    pos_dice_sum, pos_dice_n = 0.0, 0
    for img, mask, _class_label in tqdm(loader, desc="val[seg]", leave=False):
        img, mask = img.to(device), mask.to(device)
        out = model(img)
        pred_mask = out["seg"] if isinstance(out, dict) else out
        loss = loss_fn(pred_mask, mask).mean()
        running_loss += loss.item()
        running_dice += dice_score(pred_mask, mask)
        d_sum, d_n = dice_score_positive_sum(pred_mask, mask)
        pos_dice_sum += d_sum
        pos_dice_n += d_n
    n = len(loader)
    return {
        "loss": running_loss / n,
        "dice": running_dice / n,
        "pos_dice": pos_dice_sum / max(pos_dice_n, 1),
    }


def train_one_epoch_detector(model, loader, optimizer, device=config.DEVICE, scaler=None):
    model.train()
    yolo_loss_fn = YoloLoss()
    running_loss = 0.0

    for img, target, _class_label, _boxes in tqdm(loader, desc="train[det]", leave=False):
        img, target = img.to(device), target.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                             enabled=(scaler is not None)):
            out = model(img)
            pred = out["det"] if isinstance(out, dict) else out
            loss = yolo_loss_fn(pred, target).mean()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return {"loss": running_loss / len(loader)}


@torch.no_grad()
def evaluate_detector(model, loader, device=config.DEVICE):
    model.eval()
    yolo_loss_fn = YoloLoss()
    running_loss = 0.0
    for img, target, _class_label, _boxes in tqdm(loader, desc="val[det]", leave=False):
        img, target = img.to(device), target.to(device)
        out = model(img)
        pred = out["det"] if isinstance(out, dict) else out
        loss = yolo_loss_fn(pred, target).mean()
        running_loss += loss.item()
    return {"loss": running_loss / len(loader)}


def train_one_epoch_classifier(model, loader, optimizer, device=config.DEVICE, scaler=None):
    model.train()
    ce = torch.nn.CrossEntropyLoss()
    running_loss, running_acc = 0.0, 0.0

    for img, label in tqdm(loader, desc="train[cls]", leave=False):
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                             enabled=(scaler is not None)):
            out = model(img)
            logits = out["class"] if isinstance(out, dict) else out
            loss = ce(logits, label)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        running_acc += classification_accuracy(logits.detach(), label)

    n = len(loader)
    return {"loss": running_loss / n, "acc": running_acc / n}


@torch.no_grad()
def evaluate_classifier(model, loader, device=config.DEVICE):
    model.eval()
    ce = torch.nn.CrossEntropyLoss()
    running_loss, running_acc = 0.0, 0.0
    for img, label in tqdm(loader, desc="val[cls]", leave=False):
        img, label = img.to(device), label.to(device)
        out = model(img)
        logits = out["class"] if isinstance(out, dict) else out
        loss = ce(logits, label)
        running_loss += loss.item()
        running_acc += classification_accuracy(logits, label)
    n = len(loader)
    return {"loss": running_loss / n, "acc": running_acc / n}


# ---------------------------------------------------------------------------
# Joint MTL training (used by the 04 notebook)
# ---------------------------------------------------------------------------
def train_one_epoch_mtl(model, loader, optimizer, device=config.DEVICE,
                         loss_weights=(1.0, 1.0, 1.0), scaler=None, pos_weight="auto"):
    """loss_weights = (classification_w, detection_w, segmentation_w)"""
    model.train()
    ce = torch.nn.CrossEntropyLoss()
    yolo_loss_fn = YoloLoss()
    seg_loss_fn = BCEDiceLoss(pos_weight=pos_weight)
    w_cls, w_det, w_seg = loss_weights

    totals = {"loss": 0.0, "cls_loss": 0.0, "det_loss": 0.0, "seg_loss": 0.0}
    pos_dice_sum, pos_dice_n = 0.0, 0

    for batch in tqdm(loader, desc="train[mtl]", leave=False):
        img = batch["image"].to(device)
        class_label = batch["class"].to(device)
        seg_target = batch["seg"].to(device)
        det_target = batch["det"].to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                             enabled=(scaler is not None)):
            out = model(img)
            cls_loss = ce(out["class"], class_label)
            det_loss = yolo_loss_fn(out["det"], det_target).mean()
            seg_loss = seg_loss_fn(out["seg"], seg_target).mean()
            loss = w_cls * cls_loss + w_det * det_loss + w_seg * seg_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        totals["loss"] += loss.item()
        totals["cls_loss"] += cls_loss.item()
        totals["det_loss"] += det_loss.item()
        totals["seg_loss"] += seg_loss.item()
        d_sum, d_n = dice_score_positive_sum(out["seg"].detach(), seg_target)
        pos_dice_sum += d_sum
        pos_dice_n += d_n

    n = len(loader)
    result = {k: v / n for k, v in totals.items()}
    result["pos_dice"] = pos_dice_sum / max(pos_dice_n, 1)
    return result


@torch.no_grad()
def evaluate_mtl(model, loader, device=config.DEVICE, loss_weights=(1.0, 1.0, 1.0),
                  pos_weight="auto"):
    model.eval()
    ce = torch.nn.CrossEntropyLoss()
    yolo_loss_fn = YoloLoss()
    seg_loss_fn = BCEDiceLoss(pos_weight=pos_weight)
    w_cls, w_det, w_seg = loss_weights

    totals = {"loss": 0.0, "cls_loss": 0.0, "det_loss": 0.0, "seg_loss": 0.0, "acc": 0.0}
    pos_dice_sum, pos_dice_n = 0.0, 0

    for batch in tqdm(loader, desc="val[mtl]", leave=False):
        img = batch["image"].to(device)
        class_label = batch["class"].to(device)
        seg_target = batch["seg"].to(device)
        det_target = batch["det"].to(device)

        out = model(img)
        cls_loss = ce(out["class"], class_label)
        det_loss = yolo_loss_fn(out["det"], det_target).mean()
        seg_loss = seg_loss_fn(out["seg"], seg_target).mean()
        loss = w_cls * cls_loss + w_det * det_loss + w_seg * seg_loss

        totals["loss"] += loss.item()
        totals["cls_loss"] += cls_loss.item()
        totals["det_loss"] += det_loss.item()
        totals["seg_loss"] += seg_loss.item()
        totals["acc"] += classification_accuracy(out["class"], class_label)
        d_sum, d_n = dice_score_positive_sum(out["seg"], seg_target)
        pos_dice_sum += d_sum
        pos_dice_n += d_n

    n = len(loader)
    result = {k: v / n for k, v in totals.items()}
    result["pos_dice"] = pos_dice_sum / max(pos_dice_n, 1)
    return result


def fit(train_fn, eval_fn, model, train_loader, val_loader, optimizer, epochs,
        device=config.DEVICE, checkpoint_path=None, monitor="loss", mode="min",
        patience=None, scheduler=None, **kwargs):
    """Generic training driver: runs `epochs` epochs, tracks history, saves
    the best checkpoint (by `monitor`, using train_fn/eval_fn signatures
    above).

    patience: if set, stop once `monitor` hasn't improved for this many
        consecutive epochs (fixes the detector/classifier notebooks running
        all epochs and overfitting well past their best checkpoint).
    scheduler: optional torch LR scheduler. `ReduceLROnPlateau` is stepped
        with the monitored val score; any other scheduler is stepped once
        per epoch with no arguments.
    """
    import inspect

    eval_params = set(inspect.signature(eval_fn).parameters)
    eval_kwargs = {k: v for k, v in kwargs.items() if k in eval_params}

    history = {}
    best_score = float("inf") if mode == "min" else -float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_metrics = train_fn(model, train_loader, optimizer, device=device, **kwargs)
        val_metrics = eval_fn(model, val_loader, device=device, **eval_kwargs)

        print(f"Epoch {epoch}/{epochs} | "
              + " ".join(f"train_{k}={v:.4f}" for k, v in train_metrics.items())
              + " | "
              + " ".join(f"val_{k}={v:.4f}" for k, v in val_metrics.items()))

        for k, v in train_metrics.items():
            history.setdefault(f"train_{k}", []).append(v)
        for k, v in val_metrics.items():
            history.setdefault(f"val_{k}", []).append(v)

        score = val_metrics.get(monitor, val_metrics.get("loss"))
        is_better = score < best_score if mode == "min" else score > best_score
        if is_better:
            best_score = score
            epochs_without_improvement = 0
            if checkpoint_path:
                model.save(checkpoint_path)
                print(f"  -> saved new best checkpoint ({monitor}={score:.4f})")
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()

        if patience is not None and epochs_without_improvement >= patience:
            print(f"  -> early stopping: no improvement in '{monitor}' for "
                  f"{patience} epochs (best={best_score:.4f})")
            break

    return history
