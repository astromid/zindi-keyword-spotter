import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from scipy.io import wavfile
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
        sample_rate: int,
        file_paths: Optional[Sequence[Path]] = None,
        labels: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.pad_length = pad_length
        self.sample_rate = sample_rate
        self.file_paths = file_paths
        self.labels = labels
        self.le = None
        self.label_idxs = None
        
        if self.labels is not None:
            self.le = LabelEncoder()
            self.le.fit(sorted(self.labels))
            self.label_idxs = self.le.transform(self.labels)
        
        self.samples = []
        for path in tqdm(self.file_paths, desc='Loading files'):
            rate, sample = wavfile.read(path)
            self.samples.append(sample)

            if rate != self.sample_rate:
                warnings.warn(f'File {path.name} has different sample rate {rate}')

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int):
        sample = pad_sample(self.samples[index], self.pad_length)
        sample_tensor = torch.from_numpy(sample)
        if self.label_idxs is not None:
            return sample_tensor, self.label_idxs[index]
        else:
            return sample_tensor
