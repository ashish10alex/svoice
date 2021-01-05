"""
File to separate using checkpoint saved by pytorch_lightning
"""

import torch
from svoice.models.swave import SWave
import hydra
from pl_train import Model
import librosa
from svoice.separate import write
from dataset import LibriMix
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.metrics import get_metrics
from tqdm import tqdm
import pandas as pd
from pprint import pprint
import os
import json
from asteroid.utils import tensors_to_device
from timit_ss_dataset import Timit_Noisy
import argparse

parser = argparse.ArgumentParser()

#parser.add_argument(
#    "--model_path", type=str, required=True, help="path to separation model"
#)
#parser.add_argument(
#    "--mix_path", type=str, required=True, help="path to mixtures of different snrs or clean"
#)


checkpoint_path = "/jmain01/home/JAD007/txk02/aaa18-txk02/SVOICE_exp/svoice_og/outputs/exp_/lightning_logs/version_542690/checkpoints/epoch=23-step=726023.ckpt"
compute_metrics = ["si_sdr"]


@hydra.main(config_path="conf", config_name="config.yaml")
def separate_pl(args, path=''):
    mix_json_paths = dict(args['tests']['mix_paths'])
    for name, path in zip(mix_json_paths.keys(), mix_json_paths.values()):
        print('test case: ', name)
        with torch.no_grad():
            # load model
            kwargs = dict(args.swave)
            kwargs["sr"] = args.sample_rate
            kwargs["segment"] = args.segment
            model_klass = SWave(**kwargs)
            trained_model = Model.load_from_checkpoint(
                checkpoint_path, model=model_klass, args=args
            )

            trained_model.to(args.device)

            # loss function - used to reorder the sources wrt to ground truth sources
            loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
            
            
            #unseeen test dataset object
            tt_data = Timit_Noisy(
                    mix_json= str(path),
                    s1_json='/jmain01/home/JAD007/txk02/aaa18-txk02/Conv-TasNet/src/data_json/test_aug_noise/s1.json',
                    s2_json='/jmain01/home/JAD007/txk02/aaa18-txk02/Conv-TasNet/src/data_json/test_aug_noise/s2.json',
                    sample_rate=8000, 
                    n_src=2,
                    segment=None)

            series_list = []
            for idx in tqdm(range(len(tt_data))):
                #import pdb; pdb.set_trace()
                mix, sources = tensors_to_device(tt_data[idx], device=args.device)
                est_sources = trained_model(mix[None])[-1]

                loss, reordered_sources = loss_func(
                    est_sources, sources[None], return_est=True
                )

                mix_np = mix.cpu().data.numpy()
                sources_np = sources.cpu().data.numpy()
                est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()

                utt_metrics = get_metrics(
                    mix_np,
                    sources_np,
                    est_sources_np,
                    sample_rate=8000,
                    metrics_list=compute_metrics,
                )

                utt_metrics["mix_path"] = tt_data.mixture_path
                series_list.append(pd.Series(utt_metrics))
                all_metrics_df = pd.DataFrame(series_list)
                #if idx == 5:
                #    break

            all_metrics_df = pd.DataFrame(series_list)
            os.makedirs('all_metrics', exist_ok=True)
            all_metrics_df.to_csv("all_metrics/{}.csv".format(name))

            final_results = {}
            for metric_name in compute_metrics:
                input_metric_name = "input_" + metric_name
                ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
                final_results[metric_name] = all_metrics_df[metric_name].mean()
                final_results[metric_name + "_imp"] = ldf.mean()

            print("Overall metrics for {} : ".format(name))
            pprint(final_results)
            
            os.makedirs('final_metrics', exist_ok=True)
            with open(os.path.join("final_metrics/{}.json".format(name)), "w") as f:
                json.dump(final_results, f, indent=0)

if __name__== "__main__":
    separate_pl()
