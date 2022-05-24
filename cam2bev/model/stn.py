import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as functional

from . import common


class SpatialTransformer(nn.Module):

    def __init__(
            self,
            n_heads: int,
            in_channels: int,
            intermediate_channels: int = 48,
            use_dropout: bool = False,
            kernel_size: int = 3,
    ):
        super(SpatialTransformer, self).__init__()

        self._n_heads = n_heads
        self._in_ch = in_channels
        self._k_size = kernel_size
        self._dropout = use_dropout
        self._inter = intermediate_channels

        conv_modules = [
            common.conv_3x3_block(self._inter, self._inter, bn=False, relu=True, pool=True)
            for _ in range(3)
        ]

        self._conv = nn.Sequential(
            common.conv_3x3_block(self._in_ch, self._inter, bn=False, relu=True, pool=True),
            *conv_modules
        )

        self._attn_extractor = nn.Sequential(
            nn.Conv2d(in_channels=self._inter, out_channels=n_heads, kernel_size=1),
            nn.Softmax2d(),
        )

        self._fc1 = nn.Linear(self._inter, 4 * self._inter)
        self._fc2 = nn.Linear(4 * self._inter, 6)

    def forward(
            self,
            x: torch.Tensor,
            size: t.Optional[t.Tuple[int, int]] = None,
    ) -> torch.Tensor:
        batch_images = x

        x = self._conv(x.detach())
        attn_map = self._attn_extractor(x)  # B, Heads, H, W

        heads: t.List[torch.Tensor] = torch.split(attn_map, 1, dim=1)

        features = []
        for head in heads:
            features.append(torch.sum(x * head, dim=(2, 3)))  # functional is B, inter

        resampled_rois = []

        for feature in features:
            x = feature
            x = x.view(-1, self._inter)
            if self._dropout:
                x = functional.dropout(self._fc1(x), p=0.5)
                x = functional.dropout(self._fc2(x), p=0.5)
            else:
                x = self._fc1(x)
                x = self._fc2(x)  # params [Nx6]

            x = x.view(-1, 2, 3)  # change it to the 2x3 matrix

            if size is not None:
                grid_size = [x.size(0), self._in_ch, *size]
            else:
                grid_size = list(batch_images.shape)

            affine_grid_points = functional.affine_grid(x, grid_size)
            rois = functional.grid_sample(batch_images, affine_grid_points)

            resampled_rois.append(rois)

        return torch.cat(resampled_rois, dim=1)


def test_usage() -> torch.Tensor:
    stn = SpatialTransformer(n_heads=2, in_channels=3)
    return stn.forward(torch.randn(1, 3, 256, 256), size=(100, 100))  # -> (1, 3, 100, 100)
