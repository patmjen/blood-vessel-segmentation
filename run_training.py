import os
from os.path import join
from argparse import ArgumentParser
import datetime

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint

import vnet
import unet
import cli

def main(hparams):
    today = datetime.datetime.now().strftime('%d.%m.%Y')
    checkpoint_callback = ModelCheckpoint(
        dirpath=join(hparams.logger_save_dir, hparams.experiment_name, 'ckpts'),
        filename='ckpt-' + today + '-{epoch:02d}-{val_loss:2f}',
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.monitor_loss,
        prefix='')

    tb_logger = loggers.TensorBoardLogger(save_dir=hparams.logger_save_dir,
                                          name=hparams.experiment_name)
    if hparams.checkpoint_path is None:
        model = hparams.Model(**vars(hparams))
    else:
        # If any arguments were explicitly given, then force those
        seen_params = { a : getattr(hparams, a) for a in hparams.seen_args_
                        if a != '==SUPPRESS==' }
        model = hparams.Model.load_from_checkpoint(hparams.checkpoint_path,
                                                   **seen_params)

    trainer = Trainer.from_argparse_args(
        hparams,
        callbacks=[checkpoint_callback],
        logger=tb_logger)

    trainer.fit(model)


if __name__ == '__main__':
    now = datetime.datetime.now()
    dt_str = now.strftime("%d-%m-%Y_%H-%M-%S")

    parser = ArgumentParser()
    parser.add_argument('--logger_save_dir', default='D:/tmp/logs/december/')
    parser.add_argument('--monitor_loss', default='val_loss')
    parser.add_argument('--save_top_k', default=1, type=int)
    parser.add_argument('--experiment_name', default='vnet_testing_' + dt_str)
    parser.add_argument('--date_time', default=dt_str)
    parser.add_argument('--checkpoint_path', default=None)

    parser = Trainer.add_argparse_args(parser)

    subparsers = parser.add_subparsers(required=True)

    vnet_parser = vnet.VNet.add_model_specific_args(
        subparsers.add_parser('vnet'))
    unet_parser = unet.UNet3dTrainer.add_model_specific_args(
        subparsers.add_parser('unet'))

    # Override pytorch_lightning defaults
    parser.set_defaults(max_epochs=5000, gpus=1)

    cli.add_argument_tracking(vnet_parser)
    cli.add_argument_tracking(unet_parser)
    parser = cli.add_argument_tracking(parser, extra_name='seen_prog_args_')

    hparams = parser.parse_args()

    main(hparams)
