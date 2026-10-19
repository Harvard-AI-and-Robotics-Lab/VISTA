# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from Nokia Bell Labs PRIMUS; see LICENSES.txt.
"""PRIMUS multimodal IMU encoder with checkpoint-compatible tensor layouts.

The GRU receives tensors shaped [batch, 32, 32], producing a representation of
an entire IMU window. The encoder uses the multimodal head; checkpoint conversion
excludes the self-supervised head."""

from torch import nn


class Block(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, adaptive=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                input_channels, output_channels, kernel_size, dilation=2, bias=False
            ),
            nn.AdaptiveAvgPool1d(32) if adaptive else nn.MaxPool1d(3),
        )

    def forward(self, values):
        return self.net(values)


class PrimusEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.GroupNorm(2, 6),
            Block(6, 32, 10),
            Block(32, 32, 5),
            Block(32, 32, 5, adaptive=True),
            nn.GroupNorm(4, 32),
            nn.GRU(batch_first=True, input_size=32, hidden_size=512),
        )
        self.mmcl_head = nn.Linear(512, 512)

    def forward(self, values):
        if values.shape[-2:] != (6, 250):
            raise ValueError(
                "PRIMUS expects accel xyz + gyro xyz, 250 samples at 50 Hz"
            )
        return self.mmcl_head(self.backbone(values)[1][0])
