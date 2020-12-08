import os
import datetime
from argparse import ArgumentParser

from pytorch_lightning import Trainer

import vnet
import cli


def main(hparams):
    if hparams.checkpoint_path is None:
        model = vnet.VNet(**vars(hparams))
    else:
        # If any arguments were explicitly given, then force those
        seen_params = { a : getattr(hparams, a) for a in hparams.seen_args_ }
        checkpoint_path = seen_params.pop('checkpoint_path')
        model = vnet.VNet.load_from_checkpoint(checkpoint_path, **seen_params)

    trainer = Trainer.from_argparse_args(hparams, auto_lr_find=True)

    trainer.tune(model)


if __name__ == '__main__':
    now = datetime.datetime.now()
    dt_str = now.strftime("%d-%m-%Y_%H-%M-%S")

    parser = ArgumentParser()
    parser.add_argument('--logger_save_dir', default='D:/tmp/logs/december/')
    parser.add_argument('--save_top_k', default=1, type=int)
    parser.add_argument('--experiment_name', default='vnet_tuning_' + dt_str)
    parser.add_argument('--date_time', default=dt_str)
    parser.add_argument('--checkpoint_path', default=None)

    parser = vnet.VNet.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # Override pytorch_lightning defaults
    parser.set_defaults(max_epochs=5000, gpus=1)

    parser = cli.add_argument_tracking(parser)

    hparams = parser.parse_args()

    main(hparams)
