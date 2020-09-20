from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from dataset import ZindiAudioDataset
from models import PalSolModel


class PalSolClassifier(pl.LightningModule):

    def __init__(self, num_classes: int, sample_rate: int) -> None:
        super().__init__()
        self.model = PalSolModel(num_classes, sample_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch, *args) -> pl.TrainResult:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y.long())

        result = pl.TrainResult(loss)
        result.log('train_loss', loss, on_epoch=True)
        return result
    
    def configure_optimizers(self) -> Optional[Union[Optimizer, Sequence[Optimizer], Dict, Sequence[Dict], Tuple[List, List]]]:
        return Adam(self.parameters(), lr=1e-3)


class ZindiDataModule(pl.LightningDataModule):

    def __init__(
        self,
        pad_length: int,
        sample_rate: int,
        batch_size: int,
        train_files: Optional[Sequence[Path]] = None,
        train_labels: Optional[Sequence[str]] = None,
        val_files: Optional[Sequence[Path]] = None,
        val_labels: Optional[Sequence[str]] = None,
        test_files: Optional[Sequence[Path]] = None,
    ) -> None:
        self.pad_length = pad_length
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.train_files = train_files
        self.train_labels = train_labels
        self.val_files = val_files
        self.val_labels = val_labels
        self.test_files = test_files
    
    def setup(self, stage: Optional[str]) -> None:
        if stage == 'fit' or (stage is None and self.train_files is not None):
            self.zindi_train = ZindiAudioDataset(self.pad_length, self.sample_rate, self.test_files, self.train_labels)
            if self.val_files is not None and self.val_labels is not None:
                self.zindi_val = ZindiAudioDataset(self.pad_length, self.sample_rate, self.val_files, self.val_labels)

        if stage == 'test' or (stage is None and self.test_files is not None):
            self.zindi_test = ZindiAudioDataset(self.pad_length, self.sample_rate, self.test_files)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.zindi_train, batch_size=self.batch_size)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.zindi_val, batch_size=self.batch_size)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.zindi_test, batch_size=self.batch_size)
