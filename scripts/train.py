import logging
import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.utils.class_weight import compute_class_weight
from zindi_keyword_spotter.lightning import PLClassifier, ZindiDataModule
from zindi_keyword_spotter.models import SeResNet3, WideConvolutionsModel

sns.set()
LOG = logging.getLogger(__name__)
warnings.simplefilter('ignore')


@hydra.main(config_path='../configs', config_name='model')
def main(cfg: DictConfig) -> None:
    orig_cwd = Path(hydra.utils.get_original_cwd())
    all_train_df = pd.read_csv(orig_cwd / cfg.all_train_csv)
    test_df = pd.read_csv(orig_cwd / cfg.test_csv)
    data_dir = Path(cfg.data_dir)
    label2idx = {label: idx for idx, label in enumerate(test_df.columns[1:])}

    pl.seed_everything(cfg.seed)
    datamodule = ZindiDataModule(
        cfg=cfg,
        data_dir=data_dir,
        log_dir=Path.cwd(),
        label2idx=label2idx,
        train_df=all_train_df,
        test_df=test_df,
    )
    datamodule.prepare_data()
    datamodule.setup()
    # we assume that LB set is balanced
    val_weights = compute_class_weight('balanced', np.array(list(label2idx.keys())), datamodule.val_df['label'])
    val_weights = torch.from_numpy(val_weights).float()
    if cfg.balance_weights:
        class_weights = compute_class_weight('balanced', np.array(list(label2idx.keys())), datamodule.train_df['label'])
        class_weights = torch.from_numpy(class_weights).float()
    else:
        class_weights = None

    if cfg.model == 'seresnet3':
        model = SeResNet3(
            num_classes=datamodule.train_df['label'].nunique(),
            hop_length=cfg.hop_length,
            sample_rate=cfg.sample_rate,
            n_mels=cfg.n_mels,
            n_fft=cfg.n_fft,
            power=cfg.power,
            normalize=cfg.normalize,
            use_decibels=cfg.use_decibels,
        )
    elif cfg.model == 'wconv':
        model = WideConvolutionsModel(
            num_classes=datamodule.train_df['label'].nunique(),
            hop_length=cfg.hop_length,
            sample_rate=cfg.sample_rate,
            n_mels=cfg.n_mels,
            n_fft=cfg.n_fft,
            power=cfg.power,
            normalize=cfg.normalize,
            use_decibels=cfg.use_decibels,
        )
    else:
        raise ValueError('Incorrect model.')

    loss_params = {'focal_gamma': cfg.focal_gamma}
    pl_model = PLClassifier(
        model=model,
        loss_name=cfg.loss,
        loss_params=loss_params,
        lr=cfg.lr,
        scheduler=cfg.scheduler,
        weights=class_weights,
        val_weights=val_weights,
    )
    logger = WandbLogger(
        project='zindi-keyword-spotter',
        name=Path.cwd().name,
        config=dict(cfg),
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=(Path.cwd() / 'checkpoints').as_posix(),
        filename=cfg.model + '-{epoch:02d}-{val_loss:.3f}',
        save_top_k=2,
        mode='min',
        save_last=True,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        gpus=cfg.gpus,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    if cfg.lr_find:
        lr_finder = trainer.tuner.lr_find(pl_model, train_dataloader=datamodule.train_dataloader())
        fig = lr_finder.plot(suggest=True)
        fig.savefig(Path.cwd() / 'lr_find.png')
        pl_model.lr = lr_finder.suggestion()

    trainer.fit(pl_model, datamodule)

    trainer.test()
    sub_df = pd.DataFrame(pl_model.probs)
    sub_df.columns = test_df.columns[1:]
    sub_df.insert(0, 'fn', test_df['fn'])
    sub_df.to_csv(Path.cwd() / 'submission.csv', float_format='%.8f', index=False, header=True)


if __name__ == '__main__':
    main()
