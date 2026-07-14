"""
MultiCheXNet: one shared DenseNet-121 encoder + optional
classification / detection / segmentation heads.

Equivalent to MTL_model.py in the original Keras repo, but much simpler
because in PyTorch we don't need to manually juggle layer indices - modules
are just modules.
"""

import torch
import torch.nn as nn

from .encoder import DenseNet121Encoder
from .heads import ClassifierHead, DetectorHead, SegmenterHead


class MultiCheXNet(nn.Module):
    def __init__(self, pretrained_encoder=True, add_classifier=True,
                 add_detector=True, add_segmenter=True):
        super().__init__()
        self.add_classifier = add_classifier
        self.add_detector = add_detector
        self.add_segmenter = add_segmenter

        self.encoder = DenseNet121Encoder(pretrained=pretrained_encoder)

        self.classifier = ClassifierHead(self.encoder.out_channels) if add_classifier else None
        self.detector = DetectorHead(self.encoder.out_channels) if add_detector else None
        self.segmenter = SegmenterHead(self.encoder.out_channels, self.encoder.skip_channels) \
            if add_segmenter else None

    def forward(self, x):
        feat, skips = self.encoder(x)
        out = {}
        if self.add_classifier:
            out["class"] = self.classifier(feat)
        if self.add_detector:
            out["det"] = self.detector(feat)
        if self.add_segmenter:
            out["seg"] = self.segmenter(feat, skips)
        return out

    # ------------------------------------------------------------------
    # Convenience (de)serialization helpers
    # ------------------------------------------------------------------
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)
        return self


if __name__ == "__main__":
    model = MultiCheXNet(pretrained_encoder=False)
    y = model(torch.randn(2, 3, 256, 256))
    for k, v in y.items():
        print(k, v.shape)
