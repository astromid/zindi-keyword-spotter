import logging
import warnings
from pathlib import Path
import optuna

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
from zindi_keyword_spotter.models import ResNest, SeResNet3
import logging

sns.set()
LOG = logging.getLogger(__name__)
warnings.simplefilter('ignore')
logger = logging.getLogger('optuna')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('optuna.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)


def run(cfg, params) -> float:
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
        balance_sampler=cfg.balance_sampler,
        n_workers=cfg.n_workers,
        params=params
    )
    datamodule.prepare_data()
    datamodule.setup()
    train_df = datamodule.train_df

    if cfg.balance_weights:
        class_weights = compute_class_weight('balanced', np.array(list(label2idx.keys())), train_df['label'])
        class_weights = torch.from_numpy(class_weights).float()
    else:
        class_weights = None

    model = ResNest(
        num_classes=train_df['label'].nunique(),
        hop_length=int(params['hop_length']),
        sample_rate=cfg.sample_rate,
        n_mels=int(params['n_mels']),
        n_fft=int(params['n_fft']),
        power=params['power'],
        normalize=params['normalize'],
        use_decibels=params['use_decibels'],
        resnest_name=params['resnest_name'],
        pretrained=True
    )

    pl_model = PLClassifier(
        model=model,
        loss_name=params['loss'],
        lr=params['lr'],
        scheduler=cfg.scheduler,
        weights=class_weights,
    )

    # logger = WandbLogger(
    #     project='zindi-keyword-spotter',
    #     name=Path.cwd().name,
    #     config=dict(cfg),
    # )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=(Path.cwd() / 'checkpoints').as_posix(),
        filename='seresnet3-{epoch:02d}-{val_loss:.3f}',
        save_top_k=3,
        mode='min',
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        gpus=cfg.gpus,
        # logger=logger,
        callbacks=[checkpoint_callback],
    )
    if cfg.lr_find:
        lr_finder = trainer.tuner.lr_find(pl_model, train_dataloader=datamodule.train_dataloader())
        fig = lr_finder.plot(suggest=True)
        fig.savefig(Path.cwd() / 'lr_find.png')
        pl_model.lr = lr_finder.suggestion()

    trainer.fit(pl_model, datamodule)
    # print(pl_model.val_loss)

    trainer.test()
    sub_df = pd.DataFrame(pl_model.probs)
    sub_df.columns = test_df.columns[1:]
    sub_df.insert(0, 'fn', test_df['fn'])
    sub_df.to_csv(Path.cwd() / 'submission.csv', float_format='%.8f', index=False, header=True)
    logger.info(f'PARAMS: {params} \n LOSS: {pl_model.best_val_loss} \n {"="*10} \n')
    return pl_model.best_val_loss


@hydra.main(config_path='../configs', config_name='seresnet3')
def main(cfg):
    def objective(trial):
        params = {
            'hop_length': trial.suggest_discrete_uniform('hop_length', 150, 300, 20),
            'n_mels': trial.suggest_discrete_uniform('n_mels', 32, 128, 8),
            'n_fft': trial.suggest_int('n_fft', 1500, 2500),
            'power': trial.suggest_float('power', 1.5, 4.0),
            'normalize': trial.suggest_categorical('normalize', [True, False]),
            'use_decibels': trial.suggest_categorical('use_decibels', [True, False]),
            'resnest_name': trial.suggest_categorical('resnest_name', ['resnest101', 'resnest200', 'resnest269']),
            'time_shift': trial.suggest_int('time_shift', 0, 8000, 1000),
            'speed_tune': trial.suggest_float('speed_tune', 0, 0.4),
            'volume_tune': trial.suggest_float('volume_tune', 0, 0.4),
            'noise_vol': trial.suggest_float('noise_vol', 0, 0.4),
            'lr': trial.suggest_loguniform('lr', 1e-6, 1e-3),
            'loss': trial.suggest_categorical('loss', ['ce', 'focal'])
        }

        score = run(cfg, params)
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    print('best_trial: ')
    trial_ = study.best_trial
    print(trial_.values)
    print(trial_.params)


if __name__ == '__main__':
    main()
