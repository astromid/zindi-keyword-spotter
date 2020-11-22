from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.tensor import Tensor
from torch.utils.data import DataLoader, Dataset
from torchsampler import ImbalancedDatasetSampler

from zindi_keyword_spotter.dataset import ZindiAudioDataset
from zindi_keyword_spotter.focal_loss import FocalLoss


class PLClassifier(pl.LightningModule):

    def __init__(
        self,
        model: Module,
        loss_name: str,
        lr: float,
        scheduler: Optional[str],
        weights: Optional[Tensor] = None,
        val_weights: Optional[Tensor] = None,
        loss_params: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.scheduler = scheduler
        self.probs: Optional[np.ndarray] = None
        # lb metric is log loss
        self.val_criterion = CrossEntropyLoss(weight=val_weights)
        if loss_name == 'ce':
            self.criterion = CrossEntropyLoss(weight=weights)
        elif loss_name == 'focal':
            gamma = loss_params['gamma']
            self.criterion = FocalLoss(alpha=weights, gamma=gamma)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
    def training_step(self, batch, batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.val_criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx: int) -> Tensor:
        return self(batch).cpu()
    
    def test_epoch_end(self, outputs: List[np.ndarray]) -> None:
        logits = torch.vstack(outputs)
        self.probs = F.softmax(logits, dim=1).numpy()
    
    def configure_optimizers(self) -> Optional[Union[Optimizer, Sequence[Optimizer], Dict, Sequence[Dict], Tuple[List, List]]]:
        optimizer = Adam(self.parameters(), lr=self.lr)
        if self.scheduler is None:
            return optimizer
        elif self.scheduler == 'plateau':
            return {
                'optimizer': optimizer,
                'lr_scheduler': ReduceLROnPlateau(optimizer, factor=0.3, patience=10, eps=1e-5, cooldown=3, min_lr=1e-6, verbose=True),
                'monitor': 'val_loss',
            }


def callback_get_label(dataset: ZindiAudioDataset, idx: int) -> int:
    label2idx = dataset.label2idx
    return label2idx[dataset.labels[idx]]


class ZindiDataModule(pl.LightningDataModule):

    def __init__(
        self,
        cfg: DictConfig,
        data_dir: Path,
        log_dir: Path,
        label2idx: Optional[Dict[str, int]] = None,
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        balance_sampler: bool = False,
        n_workers: int = 1,
    ) -> None:
        super().__init__()
        self.pad_length = cfg.pad_length
        self.batch_size = cfg.batch_size
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.val_size = cfg.val_size
        self.train_utts = cfg.train_utts
        self.val_utts = cfg.val_utts
        self.aug_config = {
            'time_shift': cfg.time_shift,
            'speed_tune': cfg.speed_tune,
            'volume_tune': cfg.volume_tune,
            'noise_vol': cfg.noise_vol,
        }
        self.label2idx = label2idx
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = None
        self.balance_sampler = balance_sampler
        self.n_workers = n_workers

        self.train: Optional[Dataset] = None
        self.val: Optional[Dataset] = None
        self.test: Optional[Dataset] = None
    
    def prepare_data(self) -> None:
        utt_counts = self.train_df['utt_id'].value_counts()
        unique_utts = utt_counts[utt_counts == 1].index.values

        unique_utts = self.train_df[self.train_df['utt_id'].isin(unique_utts)]
        nonunique_utts = self.train_df[~self.train_df['utt_id'].isin(unique_utts)]

        train_df1, val_df1 = train_test_split(unique_utts, stratify=unique_utts['label'], test_size=self.val_size)
        train_df2 = nonunique_utts[nonunique_utts['utt_id'].isin(self.train_utts)]
        val_df2 = nonunique_utts[nonunique_utts['utt_id'].isin(self.val_utts)]

        train_df = pd.concat((train_df1, train_df2), axis=0)
        val_df = pd.concat((val_df1, val_df2), axis=0)

        all_labels = sorted(self.train_df['label'].unique())
        train_labels = sorted(train_df['label'].unique())
        if not (all_labels == train_labels):
            raise ValueError('train_df is corrupted: some labels are gone.')

        train_df.to_csv(self.log_dir / 'current_train.csv', index=False)
        val_df.to_csv(self.log_dir / 'current_val.csv', index=False)

        self.train_df = train_df
        self.val_df = val_df
    
    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or (stage is None and self.train_df is not None):
            self.train = ZindiAudioDataset(
                self.pad_length,
                self.train_df,
                data_dir=self.data_dir,
                mode='train',
                label2idx=self.label2idx,
                aug_config=self.aug_config,
            )
            if self.val_df is not None:
                self.val = ZindiAudioDataset(
                    self.pad_length,
                    self.val_df,
                    data_dir=self.data_dir,
                    mode='val',
                    label2idx=self.label2idx,
                )

        if stage == 'test' or (stage is None and self.test_df is not None):
            self.test = ZindiAudioDataset(self.pad_length, self.test_df, data_dir=self.data_dir, mode='test')
    
    def train_dataloader(self) -> DataLoader:
        sampler = ImbalancedDatasetSampler(self.train, callback_get_label=callback_get_label) if self.balance_sampler else None
        shuffle = False if self.balance_sampler else True
        return DataLoader(self.train, sampler=sampler, batch_size=self.batch_size, shuffle=shuffle, pin_memory=True, num_workers=self.n_workers)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.n_workers)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test, batch_size=1, shuffle=False)
