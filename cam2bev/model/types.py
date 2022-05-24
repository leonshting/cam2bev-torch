import typing as t

import torch


class I2IModel(t.Protocol):
    def forward(self, *tensors: t.Sequence[torch.Tensor]) -> torch.Tensor:
        pass

    @property
    def num_inputs(self) -> int:
        raise NotImplementedError
