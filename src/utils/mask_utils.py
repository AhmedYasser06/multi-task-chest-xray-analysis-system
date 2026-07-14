"""RLE <-> mask conversion utilities for the SIIM-ACR pneumothorax dataset.
Translated from data_loader/SIIM_ACR_dataloader.py (rle2mask, masks_as_image).
"""

import numpy as np


def rle2mask(rle: str, width: int, height: int) -> np.ndarray:
    """Decode a single run-length-encoded string into a (width, height) mask.
    NOTE: `width` must be the image's number of COLUMNS and `height` the
    number of ROWS - the returned array is transposed relative to the usual
    (rows, cols) image convention, so callers do `rle2mask(...).T` to get a
    normal (H, W) mask (see masks_as_image below). For the SIIM-ACR dataset
    images are always square (1024x1024) so this subtlety never matters in
    practice."""
    if rle is None or str(rle).strip() == "-1":
        return np.zeros((width, height), dtype=np.uint8)

    mask = np.zeros(width * height, dtype=np.uint8)
    array = np.asarray([int(x) for x in str(rle).split()])
    starts = array[0::2]
    lengths = array[1::2]

    pos = 0
    for start, length in zip(starts, lengths):
        pos += start
        mask[pos: pos + length] = 1
        pos += length

    return mask.reshape(width, height)


def masks_as_image(rle_list, shape):
    """OR-combine a list of RLE strings for one image into a single mask."""
    all_masks = np.zeros(shape, dtype=np.uint8)
    for rle in rle_list:
        if isinstance(rle, str) and rle.strip() != "-1":
            all_masks |= rle2mask(rle, shape[0], shape[1]).T.astype(np.uint8)
    return all_masks


def mask2rle(mask: np.ndarray) -> str:
    """Encode a binary (H,W) mask back into the SIIM-ACR RLE format (useful
    for Kaggle submission). Note SIIM-ACR uses *delta* start positions (each
    run's start is relative to the end of the previous run), matching
    rle2mask above."""
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    # NOTE: no +1 offset here - rle2mask above expects 0-indexed, delta-style
    # starts (this matches the SIIM-ACR ground-truth CSV convention).
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    starts = runs[::2].copy()
    lengths = runs[1::2] - starts
    deltas = starts.copy()
    if len(starts) > 1:
        deltas[1:] = starts[1:] - (starts[:-1] + lengths[:-1])
    out = np.empty(len(deltas) * 2, dtype=np.int64)
    out[0::2] = deltas
    out[1::2] = lengths
    return " ".join(str(x) for x in out)
