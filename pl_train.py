import os
import hydra

import numpy as np
import torch

from svoice.models.sisnr_loss import cal_loss
from svoice.models.swave import SWave
from svoice.data.data import Trainset, Validset
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = model
        self.args = args

    def forward(self, x):
        estimates = self.model(x)
        return estimates

    def common_step(self, batch, batch_idx):
        mix, lengths, sources = batch
        est_src = self.model(mix)
        loss = 0
        cnt = len(est_src)
        # apply a loss function after each layer
        with torch.autograd.set_detect_anomaly(True):
            for c_idx, est_src in enumerate(est_src):
                coeff = (c_idx + 1) * (1 / cnt)
                loss_i = 0
                # SI-SNR loss
                sisnr_loss, snr, est_src, reorder_est_src = cal_loss(
                    sources, est_src, lengths
                )
                loss += coeff * sisnr_loss
            loss /= len(est_src)
            return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        return {"val_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr, betas=(0.9, self.args.beta2)
        )
        return optimizer


@hydra.main(config_path="conf", config_name="config.yaml")
def main(args):

    # set random seed for deterministic outputs
    pl.seed_everything(args.seed)

    # get absolute path for datasets
    for key, value in args.dset.items():
        if isinstance(value, str) and key not in ["matching"]:
            args.dset[key] = hydra.utils.to_absolute_path(value)

    tr_dataset = Trainset(
        args.dset.train,
        sample_rate=args.sample_rate,
        segment=args.segment,
        stride=args.stride,
        pad=args.pad,
    )

    tr_loader = DataLoader(
        tr_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    cv_dataset = Validset(args.dset.valid)
    cv_loader = DataLoader(cv_dataset, batch_size=1, num_workers=args.num_workers)

    # define args and initialize swave model
    kwargs = dict(args.swave)
    kwargs["sr"] = args.sample_rate
    kwargs["segment"] = args.segment
    model = SWave(**kwargs)
    ss_model = Model(model=model, args=args)

    # initialize trainer
    trainer = pl.Trainer(
        gpus=4,
        distributed_backend="ddp",
        gradient_clip_val=args.max_norm,
        max_epochs=args.epochs,
    )
    trainer.fit(ss_model, tr_loader, cv_loader)


if __name__ == "__main__":
    main()
