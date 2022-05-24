import typing as t

import torch
from torch.nn import functional

import pytorch_lightning as pl

from model import types as model_types


class ModuleV1(pl.LightningModule):
    def __init__(
            self,
            model: model_types.I2IModel,
            key_order: t.Sequence[str],
            label_key: str,
    ):
        super(ModuleV1, self).__init__()

        self._model = model
        self._order = key_order

        self._label_key = label_key

    @property
    def inner_model(self):
        return self._model

    def forward(self, batch: t.Dict[str, t.Any]):
        ordered: t.List[torch.Tensor] = [batch[k] for k in self._order]
        return self._model.forward(*ordered)  # B, H, W, n_classes; logits

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-2)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.1,
        )
        return [optimizer], {'scheduler': sch, 'monitor': 'train_loss'}

    def training_step(self, train_batch, batch_idx):
        gt = torch.tensor(train_batch[self._label_key], dtype=torch.float)  # B, H, W, N
        logits = self.forward(batch=train_batch)

        loss_per_pixel = -functional.log_softmax(logits, dim=3) * gt
        # loss_weight = torch.sqrt(torch.mean(gt, dim=(0, 1, 2), keepdim=True))
        # loss = torch.mean(loss_per_pixel * loss_weight)

        loss = torch.mean(loss_per_pixel)

        return loss

    def validation_step(self, val_batch, batch_idx):
        gt = val_batch[self._label_key]  # B, H, W, N
        logits = self.forward(batch=val_batch)

        loss_per_pixel = -functional.log_softmax(logits, dim=3) * gt
        # loss_weight = torch.sqrt(torch.mean(gt, dim=(0, 1, 2), keepdim=True))
        # loss = torch.mean(loss_per_pixel * loss_weight)

        loss = torch.mean(loss_per_pixel)

        return loss
