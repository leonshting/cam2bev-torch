from . import types
from . import v1

import omegaconf


def get_model(config: omegaconf.DictConfig) -> types.I2IModel:
    if config.type == 'NUNetV1':
        return v1.NUNetV1(**config.params)

    raise NotImplementedError
