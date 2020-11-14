import logging
import os
import warnings
from pathlib import Path

import hydra
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from sklearn.model_selection import train_test_split
from zindi_keyword_spotter.lightning import PLClassifier, ZindiDataModule
from zindi_keyword_spotter.models import SeResNet3

LOG = logging.getLogger(__name__)
warnings.simplefilter('ignore')


@hydra.main(config_path='../configs', config_name='seresnet3')
def main(cfg: DictConfig) -> None:
    orig_cwd = Path(hydra.utils.get_original_cwd())
    all_train_df = pd.read_csv(orig_cwd / cfg.all_train_csv)
    all_train_df = all_train_df
    test_df = pd.read_csv(orig_cwd / cfg.test_csv)
    data_dir = Path(cfg.data_dir)

    pl.seed_everything(cfg.seed)
    train_df, val_df = train_test_split(all_train_df, stratify=all_train_df['label'], random_state=cfg.seed, test_size=cfg.val_size)

    all_labels = sorted(all_train_df['label'].unique())
    train_labels = sorted(train_df['label'].unique())
    if not (all_labels == train_labels):
        raise ValueError('train_df is corrupted: some labels are gone.')

    train_df.to_csv(Path.cwd() / 'current_train.csv', index=False)
    val_df.to_csv(Path.cwd() / 'current_val.csv', index=False)

    datamodule = ZindiDataModule(
        pad_length=cfg.pad_length,
        batch_size=cfg.batch_size,
        data_dir=data_dir,
        n_threads=cfg.n_threads,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    model = SeResNet3(
        num_classes=train_df['label'].nunique(),
        hop_length=cfg.hop_length,
        sample_rate=cfg.sample_rate,
        n_mels=cfg.n_mels,
        n_fft=cfg.n_fft,
        power=cfg.power,
        normalize=cfg.normalize,
        use_decibels=cfg.use_decibels,
    )

    pl_model = PLClassifier(model)

    logger = NeptuneLogger(
        api_key=os.environ['NEPTUNE_API_TOKEN'],
        project_name='astromid/zindi-keyword-spotter',
        experiment_name=Path.cwd().name,
        params=dict(cfg),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=(Path.cwd() / 'checkpoints').as_posix(),
        filename='seresnet3-{epoch:02d}-{val_loss:.3f}',
        save_top_k=3,
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        gpus=1,
        # distributed_backend='ddp',
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(pl_model, datamodule)

    trainer.test()
    sub_df = pd.DataFrame(pl_model.probs_matrix)
    sub_df.columns = datamodule.train.le.classes_
    sub_df.insert(0, 'fn', test_df['fn'])
    sub_df.to_csv(Path.cwd() / 'submission.csv', float_format='%.8f', index=False, header=True)


if __name__ == '__main__':
    main()
