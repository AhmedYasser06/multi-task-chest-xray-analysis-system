"""Plotting helpers for notebooks: overlay masks / boxes / class predictions."""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def show_image_mask(image, mask=None, pred_mask=None, title=None, figsize=(12, 4)):
    """image: HxWx3 uint8/float. mask / pred_mask: HxW in [0,1]."""
    n = 1 + int(mask is not None) + int(pred_mask is not None)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("image")
    axes[0].axis("off")

    idx = 1
    if mask is not None:
        axes[idx].imshow(image, cmap="gray")
        axes[idx].imshow(mask, cmap="Reds", alpha=0.4)
        axes[idx].set_title("ground truth mask")
        axes[idx].axis("off")
        idx += 1
    if pred_mask is not None:
        axes[idx].imshow(image, cmap="gray")
        axes[idx].imshow(pred_mask, cmap="Reds", alpha=0.4)
        axes[idx].set_title("predicted mask")
        axes[idx].axis("off")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def show_image_boxes(image, boxes, scores=None, class_names=None, class_ids=None,
                      title=None, figsize=(6, 6)):
    """boxes: list/array of [x1,y1,x2,y2]."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image, cmap="gray")
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        label = ""
        if scores is not None:
            label += f"{scores[i]:.2f}"
        if class_names is not None and class_ids is not None:
            label = f"{class_names[class_ids[i]]} {label}"
        if label:
            ax.text(x1, max(y1 - 5, 0), label, color="yellow", fontsize=10,
                    backgroundcolor="black")
    ax.axis("off")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def show_segmentation_grid(images, gt_masks, pred_masks, n_cols=4, figsize_per_cell=3.0,
                            threshold=0.5):
    """Grid of images with ground-truth (green) and predicted (blue) mask
    contours overlaid on the same panel - matches the "Segmentation
    results" example-results style in the upstream repo's README
    (coursat-ai/MultiCheXNet).

    images: list of HxWx3 arrays. gt_masks/pred_masks: list of HxW arrays
    in [0,1] (pred_masks are thresholded at `threshold` before contouring).
    """
    n = len(images)
    n_cols = min(n_cols, n) or 1
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(figsize_per_cell * n_cols, figsize_per_cell * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)

    for i in range(len(axes)):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(images[i], cmap="gray")
        if gt_masks[i] is not None and gt_masks[i].max() > 0:
            ax.contour(gt_masks[i] > threshold, colors="lime", linewidths=1.5)
        if pred_masks[i] is not None and pred_masks[i].max() > threshold:
            ax.contour(pred_masks[i] > threshold, colors="blue", linewidths=1.5)
        pred_line = plt.Line2D([0], [0], color="blue", lw=1.5, label="prediction")
        gt_line = plt.Line2D([0], [0], color="lime", lw=1.5, label="ground truth")
        ax.legend(handles=[pred_line, gt_line], loc="lower right", fontsize=6,
                  framealpha=0.6)
    plt.tight_layout()
    plt.show()


def show_detection_grid(images, gt_boxes_list, pred_boxes_list, pred_scores_list=None,
                         n_cols=4, figsize_per_cell=3.0):
    """Grid of images with ground-truth (green) and predicted (blue) boxes
    overlaid on the same panel - matches the "Detection results"
    example-results style in the upstream repo's README
    (coursat-ai/MultiCheXNet).

    images: list of HxWx3 arrays. gt_boxes_list/pred_boxes_list: list of
    box-list-per-image, each box as [x1,y1,x2,y2].
    """
    n = len(images)
    n_cols = min(n_cols, n) or 1
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(figsize_per_cell * n_cols, figsize_per_cell * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)

    for i in range(len(axes)):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(images[i], cmap="gray")
        for box in gt_boxes_list[i]:
            x1, y1, x2, y2 = box
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5,
                                            edgecolor="lime", facecolor="none"))
        for j, box in enumerate(pred_boxes_list[i]):
            x1, y1, x2, y2 = box
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5,
                                            edgecolor="blue", facecolor="none"))
            if pred_scores_list is not None:
                ax.text(x1, max(y1 - 4, 0), f"{pred_scores_list[i][j]:.2f}",
                        color="yellow", fontsize=7, backgroundcolor="black")
        pred_line = plt.Line2D([0], [0], color="blue", lw=1.5, label="prediction")
        gt_line = plt.Line2D([0], [0], color="lime", lw=1.5, label="ground truth")
        ax.legend(handles=[pred_line, gt_line], loc="lower right", fontsize=6,
                  framealpha=0.6)
    plt.tight_layout()
    plt.show()


def show_training_curves(history, keys=None, figsize=(14, 4)):
    """history: dict of {metric_name: [values per epoch]}."""
    keys = keys or list(history.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=figsize)
    if len(keys) == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        ax.plot(history[k])
        ax.set_title(k)
        ax.set_xlabel("epoch")
    plt.tight_layout()
    plt.show()
