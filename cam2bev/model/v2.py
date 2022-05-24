import typing as t

import torch
import torch.nn as nn

from timm.models import convnext


from . import common
from . import types
from . import stn


class BasicUNet(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, bilinear=False):
        super(BasicUNet, self).__init__()
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.bilinear = bilinear

        self.inc = common.DoubleConv(n_in_channels, 64)
        self.down1 = common.Down(64, 128)
        self.down2 = common.Down(128, 256)
        self.down3 = common.Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = common.Down(512, 1024 // factor)
        self.up1 = common.Up(1024, 512 // factor, bilinear)
        self.up2 = common.Up(512, 256 // factor, bilinear)
        self.up3 = common.Up(256, 128 // factor, bilinear)
        self.up4 = common.Up(128, 64, bilinear)

        self.out = common.OutConv(64, n_out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)


class UNetV2(nn.Module):
    def __init__(
            self,
            n_in_channels: int,
            n_out_channels: int,
            feature_dim: int = 256,
    ):
        super(UNetV2, self).__init__()

        self.stn = stn.SpatialTransformerV2(
            in_channels=feature_dim,
            intermediate_channels=feature_dim,
            use_dropout=True,
        )

        self.extractor = convnext.ConvNeXt(
            in_chans=n_in_channels,
            num_classes=feature_dim,
        )

        self.seg = BasicUNet(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
        )

    def forward(self, x) -> torch.Tensor:
        feature = self.extractor(x)
        transformed = self.stn(x, feature)
        unet = self.seg(transformed)
        return unet


class NUNetV2(nn.Module, types.I2IModel):
    def __init__(
            self,
            n_inputs: int = 4,
            n_classes: int = 5,
            n_in_channels: int = 3,
            n_out_channels: int = 64,
            stn_feature_dim: int = 256,
    ):
        super(NUNetV2, self).__init__()

        self._num_inputs = n_inputs

        for i in range(n_inputs):
            self.add_module(
                f'unet_{i}',
                UNetV2(
                    n_in_channels=n_in_channels,
                    n_out_channels=n_out_channels,
                    feature_dim=stn_feature_dim,
                )
            )

        self.outc = common.OutConv(n_inputs * n_out_channels, n_classes)

    def forward(self, *per_cam_batches: t.Sequence[torch.Tensor]):
        features = []

        for num, batch in enumerate(per_cam_batches):
            mod: nn.Module = getattr(self, f'unet_{num}')
            features.append(mod(batch))

        return self.outc(torch.cat(features, dim=1))

    @property
    def num_inputs(self) -> int:
        return self._num_inputs


def test_usage() -> torch.Tensor:
    unet = UNetV2(n_in_channels=3, n_out_channels=64)
    return unet.forward(torch.randn(1, 3, 256, 256))


def test_composed_usage() -> torch.Tensor:
    unet = NUNetV2(n_inputs=4, n_classes=5)

    tensors = [torch.randn(1, 3, 256, 256) for _ in range(4)]
    return unet.forward(*tensors)
