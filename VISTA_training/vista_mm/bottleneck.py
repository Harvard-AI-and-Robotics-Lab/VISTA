# Copyright 2026 The Scenic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synchronous multimodal bottleneck exchange with validity-aware routing.

Each sensor stream exchanges information through shared bottleneck states.
Routing weights combine modality updates with an identity fallback. The uniform
routing mode averages the available streams.

Based on Scenic's Multimodal Bottleneck Transformer (MBT). See LICENSES.txt for
upstream attribution and license terms."""

import torch
from torch import nn

from .config import SensorConfig


class BottleneckBlock(nn.Module):
    """Each stream sees the same incoming bottleneck; stream order is immaterial.

    Inputs: tokens [B,M,K,D], bottleneck [B,L,D], present [B,M], mixture [B,M+1].
    The last mixture weight retains the incoming bottleneck. No cross-modal token
    attention occurs outside these L shared states. Encoder weights are per modality,
    following the authors' share_encoder=False setting.
    """

    def __init__(self, config: SensorConfig):
        super().__init__()
        self.modalities = tuple(config.modalities)
        self.encoders = nn.ModuleDict(
            {
                modality: nn.TransformerEncoderLayer(
                    d_model=config.width,
                    nhead=config.heads,
                    dim_feedforward=4 * config.width,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for modality in self.modalities
            }
        )

    def forward(self, tokens, bottleneck, present, mixture):
        count = tokens.shape[2]
        next_tokens, candidates = [], []
        bottleneck_valid = torch.ones(
            bottleneck.shape[:2], device=bottleneck.device, dtype=torch.bool
        )
        for index, modality in enumerate(self.modalities):
            available = present[:, index, None, None]
            stream = tokens[:, index].masked_fill(~available, 0)
            combined = torch.cat([stream, bottleneck], dim=1)
            valid = torch.cat(
                [present[:, index, None].expand(-1, count), bottleneck_valid], dim=1
            )
            encoded = self.encoders[modality](combined, src_key_padding_mask=~valid)
            next_tokens.append(encoded[:, :count].masked_fill(~available, 0))
            candidates.append(encoded[:, count:])
        # Aggregate only after every branch has read the SAME incoming bottleneck.
        # Float32 accumulation keeps small sensor contributions under bf16 autocast.
        candidates = torch.stack(candidates + [bottleneck], dim=1)
        updated = (candidates.float() * mixture[:, :, None, None]).sum(dim=1)
        return torch.stack(next_tokens, dim=1), updated.to(bottleneck.dtype)
