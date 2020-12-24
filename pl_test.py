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

# checkpoint_path = "/jmain01/home/JAD007/txk02/aaa18-txk02/SVOICE_exp/svoice_og/outputs/exp_/lightning_logs/version_542154/checkpoints/epoch=4-step=151254.ckpt"
checkpoint_path = "/jmain01/home/JAD007/txk02/aaa18-txk02/svoice/outputs/exp_/lightning_logs/version_0/checkpoints/epoch=06-val_loss=26.73.ckpt"
compute_metrics = ["si_sdr"]


@hydra.main(config_path="conf", config_name="config.yaml")
def separate_pl(args):
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
        # test dataset object
        tt_data = LibriMix(
            mix_dir="/jmain01/home/JAD007/txk02/aaa18-txk02/svoice/egs/librimix_dataset/tt/"
        )

        series_list = []
        for idx in tqdm(range(len(tt_data))):
            mix = tt_data[idx][0].to(args.device)
            sources = tt_data[idx][1].to(args.device)
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
            if idx == 20:
                break

        all_metrics_df = pd.DataFrame(series_list)
        all_metrics_df.to_csv("all_metrics.csv")

        final_results = {}
        for metric_name in compute_metrics:
            input_metric_name = "input_" + metric_name
            ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
            final_results[metric_name] = all_metrics_df[metric_name].mean()
            final_results[metric_name + "_imp"] = ldf.mean()

        print("Overall metrics :")
        pprint(final_results)

        with open(os.path.join("final_metrics.json"), "w") as f:
            json.dump(final_results, f, indent=0)


separate_pl()
