from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def pad_sample(sample: np.ndarray, pad_length: int) -> np.ndarray:
    n = len(sample)
    if n < pad_length:
        return np.pad(sample, (pad_length - n, 0), constant_values=0)
    else:
        return sample[:pad_length]


class ZindiAudioDataset(Dataset):

    def __init__(
        self,
        pad_length: int,
        input_df: pd.DataFrame,
        data_dir: Path,
        mode: str = 'train',
        le: Optional[LabelEncoder] = None,
        n_threads: int = 1,
    ) -> None:
        super().__init__()
        self.pad_length = pad_length
        self.input_df = input_df
        self.data_dir = data_dir
        self.le = le
        self.label_ids = None
        self.mode = mode
        
        if self.mode == 'train':
            self.le = LabelEncoder()
            self.le.fit(sorted(self.input_df['label'].values))
        
        if self.le is not None:
            self.label_ids = self.le.transform(self.input_df['label'].values)
        
        self.samples = []
        self.sample_paths = [self.data_dir / f_path for f_path in self.input_df['fn'].values]
        with tqdm(desc=f'Loading {self.mode} samples', total=len(self.input_df)) as pbar:
            with ThreadPool(n_threads) as pool:
                for sample, _ in pool.imap(librosa.load, self.sample_paths):
                    self.samples.append(sample)
                    pbar.update()

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int):
        sample = pad_sample(self.samples[index], self.pad_length)
        sample_tensor = torch.from_numpy(sample)
        if self.label_ids is not None:
            return sample_tensor, self.label_ids[index]
        else:
            return sample_tensor
