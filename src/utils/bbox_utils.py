"""
Bounding-box helpers: convert (x1,y1,x2,y2) boxes into the YOLO grid target
tensor used by the detection head, and decode network outputs back into
boxes (with simple confidence-threshold + NMS).

Translated from data_loader/RSNA_dataloader.py (bbToYoloFormat,
findBestPrior, processGroundTruth) but written in pure numpy (no imgaug /
tensorflow dependency).
"""

import numpy as np
import torch

from .. import config


def xyxy_to_yolo(boxes):
    """(N,4) x1,y1,x2,y2 (pixels) -> (N,4) center_x,center_y,w,h (pixels)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return np.stack([cx, cy, w, h], axis=1)


def best_anchor(box_wh_grid, anchors):
    """box_wh_grid: (N,2) in grid-cell units. anchors: (A,2). -> (N,) idx."""
    w1, h1 = box_wh_grid[:, 0], box_wh_grid[:, 1]
    w2, h2 = anchors[:, 0], anchors[:, 1]
    inter = np.minimum(w1[:, None], w2) * np.minimum(h1[:, None], h2)
    union = (w1 * h1)[:, None] + (w2 * h2) - inter
    iou = inter / np.clip(union, 1e-9, None)
    return np.argmax(iou, axis=1)


def boxes_to_yolo_target(boxes_xyxy, img_size=config.IMG_SIZE,
                          grid_size=config.GRID_SIZE, anchors=config.ANCHORS,
                          n_classes=config.N_DET_CLASSES, class_ids=None):
    """
    boxes_xyxy : (N,4) array of pixel boxes for ONE image, already resized to
                 img_size x img_size. If N == 0, returns an all-zero target
                 (meaning "no object" - image has label but nothing to detect).
    class_ids  : optional (N,) int array; defaults to all zeros (single class).

    Returns a (grid_size, grid_size, n_anchors, 5+n_classes) float32 array.
    """
    stride = img_size / grid_size
    anchors = np.asarray(anchors, dtype=np.float32)
    target = np.zeros((grid_size, grid_size, len(anchors), 5 + n_classes), dtype=np.float32)

    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return target

    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
    if class_ids is None:
        class_ids = np.zeros(len(boxes_xyxy), dtype=np.int64)

    yolo_boxes = xyxy_to_yolo(boxes_xyxy) / stride           # grid-cell units
    anchor_idx = best_anchor(yolo_boxes[:, 2:4], anchors)
    grid_xy = np.floor(yolo_boxes[:, :2]).astype(np.int32)
    grid_xy[:, 0] = np.clip(grid_xy[:, 0], 0, grid_size - 1)
    grid_xy[:, 1] = np.clip(grid_xy[:, 1], 0, grid_size - 1)

    one_hot = np.zeros((len(boxes_xyxy), n_classes), dtype=np.float32)
    one_hot[np.arange(len(boxes_xyxy)), class_ids] = 1.0

    values = np.concatenate([yolo_boxes, np.ones((len(boxes_xyxy), 1), dtype=np.float32), one_hot], axis=1)

    for i in range(len(boxes_xyxy)):
        gx, gy = grid_xy[i]
        a = anchor_idx[i]
        target[gy, gx, a] = values[i]

    return target


@torch.no_grad()
def decode_predictions(det_out, img_size=config.IMG_SIZE, grid_size=config.GRID_SIZE,
                        anchors=config.ANCHORS, conf_threshold=0.3, nms_iou=0.4):
    """
    det_out: (S,S,A,5+C) tensor for a SINGLE image (raw network output, before
             sigmoid/exp). Returns a list of dicts: {box:[x1,y1,x2,y2],
             score:float, class_id:int}.
    """
    device = det_out.device
    anchors_t = torch.tensor(anchors, dtype=torch.float32, device=device)
    stride = img_size / grid_size

    xy = torch.sigmoid(det_out[..., 0:2])
    cell = torch.arange(grid_size, device=device, dtype=torch.float32)
    gx = cell.view(1, -1, 1)
    gy = cell.view(-1, 1, 1)
    x = (xy[..., 0] + gx)
    y = (xy[..., 1] + gy)

    wh = anchors_t.view(1, 1, -1, 2) * torch.exp(torch.clamp(det_out[..., 2:4], max=10.0))
    obj = torch.sigmoid(det_out[..., 4])
    cls = torch.softmax(det_out[..., 5:], dim=-1)
    cls_score, cls_id = cls.max(dim=-1)
    score = obj * cls_score

    mask = score > conf_threshold
    if mask.sum() == 0:
        return []

    cx = x[mask] * stride
    cy = y[mask] * stride
    ws = wh[..., 0][mask] * stride
    hs = wh[..., 1][mask] * stride
    scores = score[mask]
    classes = cls_id[mask]

    x1 = cx - ws / 2
    y1 = cy - hs / 2
    x2 = cx + ws / 2
    y2 = cy + hs / 2
    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    keep = torchvision_nms(boxes, scores, nms_iou)

    results = []
    for i in keep:
        results.append({
            "box": boxes[i].tolist(),
            "score": scores[i].item(),
            "class_id": classes[i].item(),
        })
    return results


def torchvision_nms(boxes, scores, iou_threshold):
    try:
        from torchvision.ops import nms
        return nms(boxes, scores, iou_threshold).tolist()
    except ImportError:
        # simple fallback NMS
        idxs = scores.argsort(descending=True).tolist()
        keep = []
        while idxs:
            cur = idxs.pop(0)
            keep.append(cur)
            idxs = [i for i in idxs if _iou(boxes[cur], boxes[i]) < iou_threshold]
        return keep


def _iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter + 1e-9
    return inter / union
