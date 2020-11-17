from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.tensor import Tensor
from torch.utils.data import DataLoader, Dataset

from zindi_keyword_spotter.dataset import ZindiAudioDataset


class PLClassifier(pl.LightningModule):

    def __init__(self, model: Module, lr: float, weights: Optional[Tensor] = None) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weights = weights
        self.probs: Optional[np.ndarray] = None
    
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
    def training_step(self, batch, batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, weight=self.weights.type_as(x))

        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, weight=self.weights.type_as(x))
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx: int) -> Tensor:
        return self(batch).cpu()
    
    def test_epoch_end(self, outputs: List[np.ndarray]) -> None:
        logits = torch.vstack(outputs)
        self.probs = F.softmax(logits, dim=1).numpy()
    
    def configure_optimizers(self) -> Optional[Union[Optimizer, Sequence[Optimizer], Dict, Sequence[Dict], Tuple[List, List]]]:
        return Adam(self.parameters(), lr=self.lr)


class ZindiDataModule(pl.LightningDataModule):

    def __init__(
        self,
        pad_length: int,
        batch_size: int,
        data_dir: Path,
        label2idx: Optional[Dict[str, int]] = None,
        train_df: Optional[pd.DataFrame] = None,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> None:
        super().__init__()
        self.pad_length = pad_length
        self.batch_size = batch_size
        self.data_dir = data_dir

        self.label2idx = label2idx
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.train: Optional[Dataset] = None
        self.val: Optional[Dataset] = None
        self.test: Optional[Dataset] = None
    
    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or (stage is None and self.train_df is not None):
            self.train = ZindiAudioDataset(self.pad_length, self.train_df, data_dir=self.data_dir, mode='train', label2idx=self.label2idx)
            if self.val_df is not None:
                self.val = ZindiAudioDataset(self.pad_length, self.val_df, data_dir=self.data_dir, mode='val', label2idx=self.label2idx)

        if stage == 'test' or (stage is None and self.test_df is not None):
            self.test = ZindiAudioDataset(self.pad_length, self.test_df, data_dir=self.data_dir, mode='test')
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, pin_memory=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, pin_memory=True)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test, batch_size=1, shuffle=False)
