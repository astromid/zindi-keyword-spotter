import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torchaudio
import librosa
from scipy.io import wavfile
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def pad_sample(sample: np.ndarray, pad_length: int) -> np.ndarray:
    delta = pad_length - len(sample)
    padded = sample
    if delta > 0:
        padded = np.pad(sample, (0, delta), mode='constant', constant_values=0)
    elif delta < 0:
        begin_idx = random.randint(0, -delta)
        padded = sample[begin_idx:begin_idx + pad_length]
    return padded


class ZindiAudioDataset(Dataset):

    def __init__(
        self,
        pad_length: int,
        input_df: pd.DataFrame,
        data_dir: Path,
        mode: str = 'train',
        label2idx: Optional[Dict[str, int]] = None
    ) -> None:
        super().__init__()
        self.pad_length = pad_length
        self.data_dir = data_dir
        self.label2idx = label2idx
        self.mode = mode

        self.sample_paths = [self.data_dir / f_path for f_path in input_df['fn'].values]
        self.labels: Optional[List[str]] = None
        if self.label2idx is not None:
            self.labels = input_df['label'].values

        self.samples: List[np.ndarray] = []
        for path in tqdm(self.sample_paths, desc=f'Loading {self.mode} samples'):
            _, sample = wavfile.read(path)
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        sample = pad_sample(self.samples[idx], self.pad_length)
        if self.label2idx is not None:
            return sample, self.label2idx[self.labels[idx]]
        else:
            return sample
