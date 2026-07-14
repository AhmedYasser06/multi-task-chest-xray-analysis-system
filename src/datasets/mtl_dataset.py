"""
Joint multi-task dataset/loader, translated from data_loader/MTL_dataloader.py.

The original Keras implementation alternates: batch 1 comes from the
segmentation generator, batch 2 comes from the detection generator, and so
on, filling in `IGNORE_VALUE` for the targets of heads that a given source
dataset can't supply. We replicate the same idea with plain PyTorch
DataLoaders and a small round-robin wrapper - simple, easy to read, and easy
to extend with more sources later.

Every batch yielded by `MTLJointLoader` is a dict:
    {
        "image":  (B,3,H,W) float tensor,
        "class":  (B,) long tensor  (0 normal / 1 pneumothorax / 2 pneumonia)
        "seg":    (B,1,H,W) float tensor, or IGNORE_VALUE-filled if this
                  batch came from the detection source
        "det":    (B,S,S,A,5+C) float tensor, or IGNORE_VALUE-filled if this
                  batch came from the segmentation source
    }
"""

import itertools

import torch
from torch.utils.data import DataLoader

from .. import config


class MTLJointLoader:
    def __init__(self, seg_dataset, det_dataset, batch_size=8, num_workers=2,
                 shuffle=True, seg_collate_fn=None, det_collate_fn=None,
                 seg_sampler=None):
        """seg_sampler: optional Sampler (e.g. WeightedRandomSampler) for the
        segmentation source. If given, it's used instead of `shuffle` for
        seg_loader - lets joint training balance positive/negative
        pneumothorax images the same way notebook 01 does standalone."""
        seg_shuffle = shuffle if seg_sampler is None else False
        self.seg_loader = DataLoader(seg_dataset, batch_size=batch_size, shuffle=seg_shuffle,
                                      sampler=seg_sampler, num_workers=num_workers,
                                      collate_fn=seg_collate_fn, drop_last=True)
        self.det_loader = DataLoader(det_dataset, batch_size=batch_size, shuffle=shuffle,
                                      num_workers=num_workers, collate_fn=det_collate_fn,
                                      drop_last=True)
        self.batch_size = batch_size
        # one "epoch" = go through the longer of the two loaders once, cycling
        # the shorter one so both sources contribute every epoch.
        self.n_batches = max(len(self.seg_loader), len(self.det_loader))

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        seg_iter = itertools.cycle(self.seg_loader)
        det_iter = itertools.cycle(self.det_loader)
        for i in range(self.n_batches):
            if i % 2 == 0:
                img, mask, class_label = next(seg_iter)
                yield self._pack(img, class_label, seg=mask, det=None)
            else:
                img, det_target, class_label, _boxes = next(det_iter)
                yield self._pack(img, class_label, seg=None, det=det_target)

    def _pack(self, img, class_label, seg=None, det=None):
        b = img.shape[0]
        if seg is None:
            seg = torch.full((b, config.SEG_N_CLASSES, config.IMG_SIZE, config.IMG_SIZE),
                              config.IGNORE_VALUE)
        if det is None:
            det = torch.full((b, config.GRID_SIZE, config.GRID_SIZE, config.N_ANCHORS,
                               5 + config.N_DET_CLASSES), config.IGNORE_VALUE)
        return {
            "image": img,
            "class": class_label if torch.is_tensor(class_label) else torch.tensor(class_label),
            "seg": seg,
            "det": det,
        }
