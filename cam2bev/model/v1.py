import typing as t

import torch
import torch.nn as nn

from . import common
from . import types
from . import stn


class UNetV1(nn.Module):
    def __init__(
            self,
            n_channels: int,
            channels: t.Sequence[int] = (64, 128, 256, 384, 512),
    ):
        super(UNetV1, self).__init__()

        self.channels = channels
        self.inc = common.DoubleConv(n_channels, channels[0])

        for num, (in_ch, out_ch) in enumerate(zip(channels, channels[1:])):
            self.add_module(
                name=f'stn_{num}',
                module=stn.SpatialTransformer(in_channels=in_ch))

            self.add_module(
                name=f'down_{num}',
                module=common.Down(in_ch, out_ch),
            )

        reversed_ch = channels[::-1]
        for num, (in_ch, out_ch) in enumerate(zip(reversed_ch, reversed_ch[1:])):
            self.add_module(
                name=f'up_{num}',
                module=common.Up(in_ch + out_ch, out_ch),
            )

    def forward(self, x) -> torch.Tensor:
        raw, resampled = [self.inc(x)], []

        for num, _ in enumerate(self.channels[1:]):
            stn_mod: stn.SpatialTransformer = getattr(self, f'stn_{num}')
            resampled.append(stn_mod.forward(raw[-1]))

            down_mod: common.Down = getattr(self, f'down_{num}')
            raw.append(down_mod.forward(raw[-1]))

        up_running = self.up_0(raw[-1], resampled[-1])
        for num, resampled_t in enumerate(reversed(resampled)):
            if num == 0:
                continue

            up_mod: common.Down = getattr(self, f'up_{num}')
            up_running = up_mod.forward(up_running, resampled_t)

        return up_running


class NUNetV1(nn.Module, types.I2IModel):
    def __init__(
            self,
            n_inputs: int = 4,
            n_classes: int = 5,
            n_input_channels: int = 3,
            channels: t.Sequence[int] = (64, 128, 256, 384, 512),
    ):
        super(NUNetV1, self).__init__()

        self._num_inputs = n_inputs

        for i in range(n_inputs):
            self.add_module(
                f'unet_{i}',
                UNetV1(n_channels=n_input_channels, channels=channels)
            )

        self.outc = common.OutConv(n_inputs * channels[0], n_classes)

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
    unet = UNetV1(n_channels=3)
    return unet.forward(torch.randn(1, 3, 256, 256))


def test_composed_usage() -> torch.Tensor:
    unet = NUNetV1(n_inputs=4, n_classes=5)

    tensors = [torch.randn(1, 3, 256, 256) for _ in range(4)]
    return unet.forward(*tensors)
