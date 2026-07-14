"""
Central configuration for MultiCheXNet (PyTorch).

Every notebook / script imports constants from here so the whole
project stays consistent (image size, anchors, class names, ...).
"""

import torch

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------
IMG_SIZE = 256                      # network input is IMG_SIZE x IMG_SIZE
N_CHANNELS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ---------------------------------------------------------------------------
# Classification head (derived 3-class problem, exactly like the original
# MultiCheXNet paper): 0 = normal, 1 = pneumothorax (SIIM-ACR), 2 = pneumonia
# (RSNA). The label is *derived* from the segmentation / detection ground
# truth, so no separate classification dataset is required for MTL training.
# ---------------------------------------------------------------------------
CLASS_NAMES = ["normal", "pneumothorax", "pneumonia"]
N_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Detection head (Tiny-YOLOv2 style, single class = "pneumonia opacity")
# ---------------------------------------------------------------------------
GRID_SIZE = IMG_SIZE // 32          # 8 for 256 input
ANCHORS = [
    (1.08, 1.19), (3.42, 4.41), (6.63, 11.38), (9.42, 5.11), (16.62, 10.52)
]                                    # widths/heights in grid-cell units
N_ANCHORS = len(ANCHORS)
N_DET_CLASSES = 1                    # single foreground class: "opacity"

# ---------------------------------------------------------------------------
# Segmentation head (pneumothorax binary mask)
# ---------------------------------------------------------------------------
SEG_N_CLASSES = 1

# ---------------------------------------------------------------------------
# Ignore value used inside a joint MTL batch: whenever a sample does not have
# a ground truth for a given head (e.g. a segmentation-only image has no
# detection boxes) its target tensor is filled with IGNORE_VALUE and the
# corresponding loss term is skipped for that sample.
# ---------------------------------------------------------------------------
IGNORE_VALUE = -1.0
