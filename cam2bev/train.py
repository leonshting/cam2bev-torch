import hydra
import omegaconf
import pytorch_lightning as pl

from torch.utils import data as td

import data
import model
import module


def get_training_module(
        inner_model: model.types.I2IModel,
        config: omegaconf.DictConfig,
) -> pl.LightningModule:
    if config.type == 'ModuleV1':
        return module.ModuleV1(
            model=inner_model,
            key_order=config.params.key_order,
            label_key=config.params.label_key,
        )

    raise NotImplementedError


@hydra.main(config_path='../configs', config_name='train_v2')
def train(config: omegaconf.DictConfig):

    train_dataset = data.get_dataset(config.data.train_dataset)
    test_dataset = data.get_dataset(config.data.val_dataset)

    assert train_dataset.num_in_channels == test_dataset.num_in_channels
    assert train_dataset.num_out_channels == test_dataset.num_out_channels

    inner = model.get_model(config.model)
    mod = get_training_module(inner_model=inner, config=config.training)

    train_dataloader = td.DataLoader(
        dataset=train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
    )

    val_dataloader = td.DataLoader(
        dataset=test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
    )

    trainer = pl.Trainer(
        gpus=config.training.gpu,
        accelerator='ddp' if len(config.training.gpu) > 1 else None,
        max_epochs=config.training.num_epochs,
        progress_bar_refresh_rate=1,
        default_root_dir=config.training.log_dir,
    )

    trainer.fit(mod, train_dataloader, val_dataloader)


if __name__ == '__main__':
    train()
