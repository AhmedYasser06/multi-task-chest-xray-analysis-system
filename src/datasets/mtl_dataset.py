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
                  batch came from a source that can't supply a segmentation
                  target
        "det":    (B,S,S,A,5+C) float tensor, or IGNORE_VALUE-filled if this
                  batch came from a source that can't supply a detection
                  target
    }

Optional third source - `cls_dataset`
--------------------------------------
`seg_dataset` is expected to be built with `only_positive=True` (see
notebook 01/04): every image in it has a real pneumothorax mask, which is
what the segmentation loss needs to avoid collapsing to a "predict nothing"
optimum (see BCEDiceLoss docstring). But that also means every batch drawn
from `seg_dataset` unconditionally has `class_label=1` (pneumothorax) - if
that's the *only* SIIM-domain data the classifier ever sees, it can learn
"this image comes from the SIIM/segmentation domain -> predict
pneumothorax" as a shortcut instead of real image features, since it never
sees a SIIM-domain image labeled anything else. That shortcut memorizes
training data fine but collapses generalization (this is exactly what
produced val_acc~42-44%, barely above chance, in a real run of this
notebook).

`cls_dataset` (optional) should be a SIIMACRDataset built from the *full*,
unfiltered dataframe (`only_positive=False`/default) - it supplies the
missing negative-class diversity. Batches from this source only contribute
a classification loss: `seg` and `det` targets are both filled with
IGNORE_VALUE (this dataset's masks are not used for segmentation - reusing
them for segmentation loss would reintroduce the original empty-mask
collapse problem, since most of `cls_dataset` is negative/empty-mask).
"""

import itertools

import torch
from torch.utils.data import DataLoader

from .. import config


class MTLJointLoader:
    def __init__(self, seg_dataset, det_dataset, cls_dataset=None, batch_size=8,
                 num_workers=2, shuffle=True, seg_collate_fn=None, det_collate_fn=None,
                 cls_collate_fn=None, seg_sampler=None):
        """seg_sampler: optional Sampler (e.g. WeightedRandomSampler) for the
        segmentation source. If given, it's used instead of `shuffle` for
        seg_loader - lets joint training balance positive/negative
        pneumothorax images the same way notebook 01 does standalone.

        cls_dataset: optional third source dedicated to classification-only
        supervision - see module docstring. If omitted, behaves exactly as
        before (2-way seg/det round robin); pass it to fix the classifier
        shortcut/generalization problem described above.
        """
        seg_shuffle = shuffle if seg_sampler is None else False
        self.seg_loader = DataLoader(seg_dataset, batch_size=batch_size, shuffle=seg_shuffle,
                                      sampler=seg_sampler, num_workers=num_workers,
                                      collate_fn=seg_collate_fn, drop_last=True)
        self.det_loader = DataLoader(det_dataset, batch_size=batch_size, shuffle=shuffle,
                                      num_workers=num_workers, collate_fn=det_collate_fn,
                                      drop_last=True)
        self.cls_loader = None
        if cls_dataset is not None:
            self.cls_loader = DataLoader(cls_dataset, batch_size=batch_size, shuffle=shuffle,
                                          num_workers=num_workers, collate_fn=cls_collate_fn,
                                          drop_last=True)
        self.batch_size = batch_size
        # one "epoch" = go through the longest loader once, cycling the
        # shorter ones so every source contributes every epoch.
        lens = [len(self.seg_loader), len(self.det_loader)]
        if self.cls_loader is not None:
            lens.append(len(self.cls_loader))
        self.n_batches = max(lens)
        self.n_sources = 3 if self.cls_loader is not None else 2

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        seg_iter = itertools.cycle(self.seg_loader)
        det_iter = itertools.cycle(self.det_loader)
        cls_iter = itertools.cycle(self.cls_loader) if self.cls_loader is not None else None
        for i in range(self.n_batches):
            source = i % self.n_sources
            if source == 0:
                img, mask, class_label = next(seg_iter)
                yield self._pack(img, class_label, seg=mask, det=None)
            elif source == 1:
                img, det_target, class_label, _boxes = next(det_iter)
                yield self._pack(img, class_label, seg=None, det=det_target)
            else:
                img, _mask, class_label = next(cls_iter)
                yield self._pack(img, class_label, seg=None, det=None)

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
