import json
from pathlib import Path
import os
import time
import hydra

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from svoice.separate import separate
from svoice.evaluate import evaluate
from svoice.models.sisnr_loss import cal_loss
from svoice.models.swave import SWave
from svoice.data.data import Trainset, Validset
from torch.utils.data import DataLoader

#from .utils import bold, copy_state, pull_metric, serialize_model, swap_state, LogProgress

import pytorch_lightning as pl
from svoice.models.swave import SWave
from svoice import distrib

class Model(pl.LightningModule):
    
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args=args
    def forward(self, x):
        estimates = self.model(x)
        return estimates

    def training_step(self, batch, batch_idx):
        mix, lengths, sources = batch
        #import pdb; pdb.set_trace()
        est_src =  self.model(mix)
        loss = 0
        cnt = len(est_src)
        # apply a loss function after each layer
        with torch.autograd.set_detect_anomaly(True):
            for c_idx, est_src in enumerate(est_src):
                coeff = ((c_idx+1)*(1/cnt))
                loss_i = 0
                # SI-SNR loss
                sisnr_loss, snr, est_src, reorder_est_src = cal_loss(
                    sources, est_src, lengths)
                loss += (coeff * sisnr_loss)
            loss /= len(est_src)
            print('loss', loss)
            return {'loss': loss}

    
    #def validation_step(self, batch, batch_idx):
    #    pass

    def configure_optimizers(self):
        print('opt')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, self.args.beta2))
        return optimizer

@hydra.main(config_path="conf", config_name='config.yaml')
def main(args):
    
    #set random seed for deterministic outputs
    torch.manual_seed(args.seed)

    #get absolute path for datasets
    for key, value in args.dset.items():
        if isinstance(value, str) and key not in ["matching"]:
            args.dset[key] = hydra.utils.to_absolute_path(value)

    tr_dataset = Trainset(
    args.dset.train, sample_rate=args.sample_rate, segment=args.segment, stride=args.stride, pad=args.pad)
    tr_loader = DataLoader(
    tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    

    #define args and initialize swave model
    kwargs = dict(args.swave)
    kwargs['sr'] = args.sample_rate
    kwargs['segment'] = args.segment
    model = SWave(**kwargs)
    ss_model = Model(model=model, args=args)
    
    #initialize trainer
    trainer = pl.Trainer(gpus=1, 
                        distributed_backend='dp',
                        gradient_clip_val=args.max_norm
            )
    trainer.fit(ss_model, tr_loader)



    print('test')

if __name__ == "__main__":
    main()
