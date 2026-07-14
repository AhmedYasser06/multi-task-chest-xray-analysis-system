"""
Shared DenseNet-121 encoder.

This is the backbone that all three heads (classification, detection,
segmentation) sit on top of. It is the PyTorch equivalent of
`utils/Encoder.py` in the original Keras project.

For a 256x256 input it produces:
    - final feature map:  (B, 1024, 8, 8)   -> used by classifier & detector
    - skip_64 (B,  64, 64, 64)  -> after the stem (pool0)
    - skip_32 (B, 128, 32, 32)  -> after transition1
    - skip_16 (B, 256, 16, 16)  -> after transition2

The three skip maps are used by the segmentation decoder (see
`heads.py :: SegmenterHead`) exactly like the skip connections used in the
original Keras Tiramisu-style decoder.
"""

import torch
import torch.nn as nn
import torchvision


class DenseNet121Encoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        densenet = torchvision.models.densenet121(weights=weights)
        features = densenet.features

        # Split the backbone into stages so we can grab skip connections.
        self.stem = nn.Sequential(
            features.conv0, features.norm0, features.relu0, features.pool0
        )                                                   # -> 64 ch, /4
        self.block1 = features.denseblock1                  # -> 256 ch, /4
        self.trans1 = features.transition1                   # -> 128 ch, /8
        self.block2 = features.denseblock2                   # -> 512 ch, /8
        self.trans2 = features.transition2                   # -> 256 ch, /16
        self.block3 = features.denseblock3                   # -> 1024 ch,/16
        self.trans3 = features.transition3                   # -> 512 ch, /32
        self.block4 = features.denseblock4                   # -> 1024 ch,/32
        self.norm5 = features.norm5
        self.final_relu = nn.ReLU(inplace=True)

        self.out_channels = 1024
        self.skip_channels = {"skip_64": 64, "skip_32": 128, "skip_16": 256}

    def forward(self, x):
        skip_64 = self.stem(x)                # (B,  64, 64, 64)
        x = self.block1(skip_64)
        skip_32 = self.trans1(x)               # (B, 128, 32, 32)
        x = self.block2(skip_32)
        skip_16 = self.trans2(x)               # (B, 256, 16, 16)
        x = self.block3(skip_16)
        x = self.trans3(x)                     # (B, 512, 8, 8)
        x = self.block4(x)
        x = self.norm5(x)
        out = self.final_relu(x)               # (B, 1024, 8, 8)

        skips = {"skip_64": skip_64, "skip_32": skip_32, "skip_16": skip_16}
        return out, skips


if __name__ == "__main__":
    enc = DenseNet121Encoder(pretrained=False)
    y, skips = enc(torch.randn(2, 3, 256, 256))
    print("encoder out:", y.shape)
    for k, v in skips.items():
        print(k, v.shape)
