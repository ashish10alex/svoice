import os
import pandas as pd
from torch.utils.data import Dataset
import pandas as pd
import soundfile as sf
import torch
import random
import numpy as np
import librosa

class Timit_Noisy(Dataset):
    """Dataset class for TIMIT noisy source separation tasks."""

    dataset_name = "TIMIT noisy"

    def __init__(self, mix_json, s1_json, s2_json, sample_rate=8000, n_src=2, segment=3):
        
        self.segment = segment
        self.sample_rate = sample_rate
        self.n_src = n_src
        self.segment=segment
        # self.seg_len = int(self.segment * self.sample_rate)
        self.seg_len=None
        
        s2_df = pd.read_json(s2_json)
        s1_df = pd.read_json(s1_json)
        mix_df = pd.read_json(mix_json)
        self.mix_df =mix_df
        self.s1_df =s1_df
        self.s2_df =s2_df
        #only odd rows are mixed with noise
        # mix_df = mix_df.iloc[1::2]
        # s1_df = s1_df.iloc[1::2]
        # s2_df = s2_df.iloc[1::2]
        
        self.mix_df.columns = ['path', 'length']
        self.s1_df.columns = ['path', 'length']
        self.s2_df.columns = ['path', 'length']
        # import pdb; pdb.set_trace()
        
        #combining all info into one dataframe
        # self.df = pd.merge(mix_df, s1_df, on='length').merge(s2_df, on='length')
        # print('success')

    def __len__(self):
        return len(self.mix_df)

    def __getitem__(self, idx):
        # Get the row in dataframe
        self.mixture_path = self.mix_df['path'].iloc[idx]
        self.s1_path = self.s1_df['path'].iloc[idx]
        self.s2_path = self.s2_df['path'].iloc[idx]
        # Read the mixture
        # print(self.mixture_path)
        # print(self.s1_path)
        # print(self.s2_path)
        
        mixture, _ = librosa.load(self.mixture_path, sr=8000)
        s1, _ = librosa.load(self.s1_path, sr=8000)
        s2, _ = librosa.load(self.s2_path, sr=8000)
        sources_list =[s1, s2]
        # Convert to torch tensor
        mixture = torch.from_numpy(mixture)
        # Stack sources
        sources = np.vstack(sources_list)
        # Convert sources to tensor
        sources = torch.from_numpy(sources)
        return mixture, sources
