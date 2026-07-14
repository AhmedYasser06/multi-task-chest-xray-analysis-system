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
