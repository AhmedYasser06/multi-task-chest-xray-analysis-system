"""
The three task-specific heads that sit on top of the shared
DenseNet-121 encoder. Equivalent to utils/Classifier.py, utils/Detector.py
and utils/Segmenter.py in the original Keras project.
"""

import torch
import torch.nn as nn

from .. import config


# ---------------------------------------------------------------------------
# 1) Classification head
# ---------------------------------------------------------------------------
class ClassifierHead(nn.Module):
    """Global-average-pool -> MLP -> softmax over N_CLASSES."""

    def __init__(self, in_channels=1024, n_classes=config.N_CLASSES):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, encoder_out):
        x = self.gap(encoder_out).flatten(1)
        return self.net(x)                      # raw logits (B, n_classes)


# ---------------------------------------------------------------------------
# 2) Detection head (Tiny-YOLOv2 style)
# ---------------------------------------------------------------------------
def conv_bn_lrelu(in_ch, out_ch, k=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )


class DetectorHead(nn.Module):
    """
    Outputs a (B, n_cells, n_cells, n_anchors, 5 + n_det_classes) tensor:
    [tx, ty, tw, th, objectness, class_logits...] per anchor per grid cell.
    """

    def __init__(self, in_channels=1024, n_anchors=config.N_ANCHORS,
                 n_det_classes=config.N_DET_CLASSES):
        super().__init__()
        self.n_anchors = n_anchors
        self.n_det_classes = n_det_classes
        n_outputs = n_anchors * (5 + n_det_classes)

        self.conv1 = conv_bn_lrelu(in_channels, 1024, k=3)
        self.conv2 = nn.Conv2d(1024, n_outputs, kernel_size=1)

    def forward(self, encoder_out):
        x = self.conv1(encoder_out)
        x = self.conv2(x)                                   # (B, n_out, H, W)
        b, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()               # (B, H, W, n_out)
        x = x.view(b, h, w, self.n_anchors, 5 + self.n_det_classes)
        return x


# ---------------------------------------------------------------------------
# 3) Segmentation head (Tiramisu-like decoder with skip connections)
# ---------------------------------------------------------------------------
class DenseConvBlock(nn.Module):
    """One 'growth' block: BN-ReLU-Conv1x1-BN-ReLU-Conv3x3, concatenated
    with its input (mirrors utils/Segmenter.py::conv_block)."""

    def __init__(self, in_ch, growth_rate=32):
        super().__init__()
        inter_ch = 4 * growth_rate
        self.net = nn.Sequential(
            nn.Dropout(0.2),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return torch.cat([x, self.net(x)], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_ch, n_layers=3, growth_rate=32):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(n_layers):
            layers.append(DenseConvBlock(ch, growth_rate))
            ch += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, x):
        return self.block(x)


class TransitionUp(nn.Module):
    """ConvTranspose2d upsample by 2, then concat with an encoder skip map."""

    def __init__(self, in_ch, out_ch, skip_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2,
                                      padding=1, output_padding=1)
        self.out_channels = out_ch + skip_ch

    def forward(self, x, skip):
        x = self.up(x)
        return torch.cat([skip, x], dim=1)


class SegmenterHead(nn.Module):
    """
    8x8x1024  --up--> 16x16 (+skip_16) --denseblock--> up--> 32x32 (+skip_32)
    --denseblock--> up--> 64x64 (+skip_64) --denseblock--> up--> 128x128
    --conv--> up--> 256x256 --1x1 conv + sigmoid--> mask
    """

    def __init__(self, encoder_out_ch=1024, skip_channels=None, blocks=(3, 3, 3)):
        super().__init__()
        skip_channels = skip_channels or {"skip_64": 64, "skip_32": 128, "skip_16": 256}

        self.tu1 = TransitionUp(encoder_out_ch, 3, skip_channels["skip_16"])
        self.db1 = DenseBlock(self.tu1.out_channels, n_layers=blocks[0])

        self.tu2 = TransitionUp(self.db1.out_channels, 3, skip_channels["skip_32"])
        self.db2 = DenseBlock(self.tu2.out_channels, n_layers=blocks[1])

        self.tu3 = TransitionUp(self.db2.out_channels, 3, skip_channels["skip_64"])
        self.db3 = DenseBlock(self.tu3.out_channels, n_layers=blocks[2])

        self.up4 = nn.ConvTranspose2d(self.db3.out_channels, 256, kernel_size=3,
                                       stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        self.up5 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2,
                                       padding=1, output_padding=1)
        self.out_conv = nn.Conv2d(256, config.SEG_N_CLASSES, kernel_size=1)

    def forward(self, encoder_out, skips):
        x = self.tu1(encoder_out, skips["skip_16"])
        x = self.db1(x)
        x = self.tu2(x, skips["skip_32"])
        x = self.db2(x)
        x = self.tu3(x, skips["skip_64"])
        x = self.db3(x)
        x = self.up4(x)
        x = self.conv4(x)
        x = self.up5(x)
        x = self.out_conv(x)
        return torch.sigmoid(x)                     # (B, 1, 256, 256)
