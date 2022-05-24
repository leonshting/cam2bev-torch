import typing as t

import omegaconf

import torch.utils.data as td

from . import datasets
from . import types


def get_dataset(config: omegaconf.DictConfig) -> t.Union[types.WithIOInfo, td.Dataset]:
    if config.type == 'MultiFolderMasks':
        return datasets.MultiFolderMasks(
            root=config.params.root,
            conversion_map=config.params.classes,
            folders=config.params.folders,
            target_size=tuple(config.params.target_size),
        )

    raise NotImplementedError

