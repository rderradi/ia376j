"""
This file runs the main training/validation loop, testing, etc... using
pytorch-lightning Trainer.

Call this script with `-h` flag to see all arguments. Some are defined on this
script and others are defined on data/model modules.
"""
import os
import random
from argparse import ArgumentParser, ArgumentTypeError

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.neptune import NeptuneLogger
from src.data.docvqa import DocVQADataModule
from src.models import LitEffNetT5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def main(hparams):

    set_seed(hparams.seed)

    checkpoint_callback = None
    if hparams.checkpoint_path:
        checkpoint_dir = os.path.dirname(
            os.path.abspath(hparams.checkpoint_path))
        print(f'Checkpoints will be saved to {checkpoint_dir}')

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            prefix=hparams.checkpoint_prefix,
            monitor=hparams.checkpoint_monitor,
            mode=hparams.checkpoint_monitor_mode,
            save_top_k=hparams.checkpoint_save_top_k,
            verbose=True,
        )

    if hparams.resume_from_checkpoint:
        print(f'Restoring checkpoint: {hparams.resume_from_checkpoint}')

    logger = NeptuneLogger(
        api_key=None,  # read from NEPTUNE_API_TOKEN environment variable
        project_name=hparams.project_name,
        experiment_name=hparams.experiment_name,
        tags=hparams.experiment_tags,
        close_after_fit=False,
        params=vars(hparams)
    )

    dm = DocVQADataModule(hparams)
    dm.setup()

    model = LitEffNetT5(hparams)

    trainer = Trainer.from_argparse_args(
        hparams,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    if hparams.do_train:
        trainer.fit(model, dm)

    if hparams.do_test:
        trainer.test(datamodule=dm)

    logger.experiment.stop()


def int_or_float(x):
    """Accepts floats and integers."""
    try:
        x = float(x)
    except ValueError:
        raise ArgumentTypeError('%r should be a number.' % x)

    if x.is_integer():
        x = int(x)

    return x


if __name__ == '__main__':

    parser = ArgumentParser(add_help=False)
    parser.add_argument('--seed', type=int, default=23)

    # trainer
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--log_gpu_memory', type=bool, default=True)
    parser.add_argument('--profiler', type=bool, default=True)
    parser.add_argument('--progress_bar_refresh_rate', type=int, default=50)
    parser.add_argument('--accumulate_grad_batches', type=int, default=16)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--val_check_interval', type=int_or_float, default=0.1)
    parser.add_argument('--limit_val_batches', type=int_or_float, default=50)
    parser.add_argument('--fast_dev_run', action='store_true')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)

    # logger
    parser.add_argument('--project_name', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--experiment_tags', nargs='*')
    parser.add_argument('--log_every_n_steps', type=int, default=50)

    # checkpoint callback
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--checkpoint_prefix', type=str, default='effnett5')
    parser.add_argument('--checkpoint_monitor', type=str, default='avg_val_f1')
    parser.add_argument('--checkpoint_monitor_mode', type=str, default='max')
    parser.add_argument('--checkpoint_save_top_k', type=int, default=10)
    parser.add_argument('--checkpoint_verbose', type=bool, default=True)

    # data module specific args
    parser = DocVQADataModule.add_module_specific_args(parser)

    # model specific args
    parser = LitEffNetT5.add_model_specific_args(parser)

    hparams = parser.parse_args()

    main(hparams)
