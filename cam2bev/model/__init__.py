from . import types
from . import v1
from . import v2

import omegaconf


def get_model(config: omegaconf.DictConfig) -> types.I2IModel:
    if config.type == 'NUNetV1':
        return v1.NUNetV1(**config.params)

    if config.type == 'NUNetV2':
        return v2.NUNetV2(**config.params)

    raise NotImplementedError
