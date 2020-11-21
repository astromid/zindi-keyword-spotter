import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from acoustics.generator import noise
from librosa.effects import time_stretch
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


def time_shift(sample: np.ndarray, max_shift: int) -> np.ndarray:
    shift = random.randint(-max_shift, max_shift)
    return np.roll(sample, shift)


def speed_tune(sample: np.ndarray, max_tune: float) -> np.ndarray:
    rate = random.uniform(1 - max_tune, 1 + max_tune)
    return time_stretch(sample, rate)


def add_noise(sample: np.ndarray, noise_sample: np.ndarray, max_vol_tune: float, max_noise_vol: float) -> np.ndarray:
    volume = random.uniform(1 - max_vol_tune, 1 + max_vol_tune)
    noise_volume = random.uniform(0, max_noise_vol)
    return volume * sample + noise_volume * noise_sample


def get_noise(length: int) -> np.ndarray:
    color = random.choice(['white', 'pink', 'blue', 'brown', 'violet'])
    return noise(length, color)


class ZindiAudioDataset(Dataset):

    def __init__(
        self,
        pad_length: int,
        input_df: pd.DataFrame,
        data_dir: Path,
        aug_config: Optional[Dict[str, Union[int, float]]] = None,
        mode: str = 'train',
        label2idx: Optional[Dict[str, int]] = None
    ) -> None:
        super().__init__()
        self.pad_length = pad_length
        self.data_dir = data_dir
        self.label2idx = label2idx
        self.aug_config = aug_config
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
        sample = self.samples[idx]
        if self.mode == 'train':
            aug_flags = [random.uniform(0, 1) for _ in range(3)]
            if aug_flags[0] < 0.5 and self.aug_config['time_shift'] != 0:
                sample = time_shift(sample, self.aug_config['time_shift'])
            if aug_flags[1] < 0.5 and self.aug_config['speed_tune'] != 0:
                sample = speed_tune(sample, self.aug_config['speed_tune'])
            if aug_flags[2] < 0.5 and self.aug_config['noise_vol'] != 0:
                noise_sample = get_noise(len(sample))
                sample = add_noise(sample, noise_sample, self.aug_config['volume_tune'], self.aug_config['noise_vol'])
        sample = pad_sample(sample, self.pad_length)
        if self.label2idx is not None:
            return sample, self.label2idx[self.labels[idx]]
        else:
            return sample
