import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as functional

from . import common


class SpatialTransformer(nn.Module):

    def __init__(
            self,
            in_channels: int,
            intermediate_channels: int = 48,
            use_dropout: bool = False,
    ):
        super(SpatialTransformer, self).__init__()

        self._in_ch = in_channels
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

        self._fc1 = nn.Linear(self._inter, 4 * self._inter)
        self._fc2_trans = nn.Linear(4 * self._inter, 2)
        self._fc2_rot = nn.Linear(4 * self._inter, 4)

    def forward(
            self,
            x: torch.Tensor,
            size: t.Optional[t.Tuple[int, int]] = None,
    ) -> torch.Tensor:
        batch_images = x
        x = self._conv(x)

        feature = functional.adaptive_max_pool2d(x, output_size=(1, 1))
        features = [feature.squeeze(-1).squeeze(-1)]

        resampled_rois = []

        for feature in features:
            x = feature
            x = x.view(-1, self._inter)
            if self._dropout:
                x = functional.dropout(self._fc1(x), p=0.5)
                rot = self._fc2_rot(x)
                trans = self._fc2_trans(x)
            else:
                x = self._fc1(x)
                rot = self._fc2_rot(x)
                trans = self._fc2_trans(x)

            rot = rot.view(-1, 2, 2)
            trans = trans.view(-1, 2, 1)

            x = torch.cat([rot, trans], dim=2)

            if size is not None:
                grid_size = [x.size(0), self._in_ch, *size]
            else:
                grid_size = list(batch_images.shape)

            affine_grid_points = functional.affine_grid(x, grid_size)
            rois = functional.grid_sample(batch_images, affine_grid_points)

            resampled_rois.append(rois)

        return torch.cat(resampled_rois, dim=1)


class SpatialTransformerV2(nn.Module):

    def __init__(
            self,
            in_channels: int,
            intermediate_channels: int = 48,
            use_dropout: bool = False,
    ):
        super(SpatialTransformerV2, self).__init__()

        self._in_ch = in_channels
        self._dropout = use_dropout
        self._inter = intermediate_channels

        self._projection = nn.Linear(in_channels, self._inter)

        self._fc1 = nn.Linear(self._inter, 4 * self._inter)
        self._fc2_trans = nn.Linear(4 * self._inter, 2)
        self._fc2_rot = nn.Linear(4 * self._inter, 4)

    def forward(
            self,
            x: torch.Tensor,
            feature: torch.Tensor,
            size: t.Optional[t.Tuple[int, int]] = None,
    ) -> torch.Tensor:

        images = x

        x = functional.relu(self._projection(feature))
        x = x.view(-1, self._inter)

        if self._dropout:
            x = functional.dropout(self._fc1(x), p=0.5)
            rot = self._fc2_rot(x)
            trans = self._fc2_trans(x)
        else:
            x = self._fc1(x)
            rot = self._fc2_rot(x)
            trans = self._fc2_trans(x)

        rot = rot.view(-1, 2, 2)
        trans = trans.view(-1, 2, 1)

        x = torch.cat([rot, trans], dim=2)

        if size is not None:
            grid_size = [*images.shape[:2], *size]
        else:
            grid_size = list(images.shape)

        affine_grid_points = functional.affine_grid(x, grid_size)
        rois = functional.grid_sample(images, affine_grid_points)
        return rois


def test_usage() -> torch.Tensor:
    stn = SpatialTransformer(in_channels=3)
    return stn.forward(torch.randn(1, 3, 256, 256), size=(100, 100))  # -> (1, 3, 100, 100)


def test_usage_v2() -> torch.Tensor:
    stn = SpatialTransformerV2(in_channels=128, intermediate_channels=128)
    transformed = stn.forward(
        x=torch.randn(1, 3, 256, 256),
        feature=torch.randn(1, 128),
        size=(100, 100)
    )  # -> (1, 3, 100, 100)

    assert transformed.shape == (1, 3, 100, 100)
    return transformed
