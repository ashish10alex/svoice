'''
File to separate using model saved by pytorch_lightning
'''

import torch
from svoice.data.data import EvalDataset
from svoice.models.swave import SWave
import hydra
from pl_train import Model
import librosa
from svoice.separate import write

#model_path ='/jmain01/home/JAD007/txk02/aaa18-txk02/svoice/outputs/exp_/lightning_logs/version_0/checkpoints/epoch=0-step=0.ckpt'
checkpoint_path = '/jmain01/home/JAD007/txk02/aaa18-txk02/SVOICE_exp/svoice_og/outputs/exp_/lightning_logs/version_542154/checkpoints/epoch=4-step=151254.ckpt'

@hydra.main(config_path="conf", config_name="config.yaml")
def separate_pl(args):
    
    #load model
    kwargs = dict(args.swave)
    kwargs["sr"] = args.sample_rate
    kwargs["segment"] = args.segment
    model_klass = SWave(**kwargs)
    trained_model = Model.load_from_checkpoint(
            checkpoint_path,
            model=model_klass,
            args=args)
    
    example, _ = librosa.load('/jmain01/home/JAD007/txk02/aaa18-txk02/svoice/dataset/debug/mix/1919-142785-0045_6319-275224-0016.wav', sr=8000)
    example = torch.tensor(example)[None]
    estimates = trained_model(example)[-1]
    
    for i in range(2):
        write(estimates[0][i].detach().numpy(), 's_{}.wav'.format(i))


separate_pl()



