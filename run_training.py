import os
from pytorch_lightning import Trainer
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
import vnet
import datetime
import pytorch_lightning as pl


def main(hparams):
    today = datetime.datetime.now().strftime('%d.%m.%Y')
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.logger_save_dir+hparams.experiment_name+'/ckpts/',
        filename='ckpt-' + today + '-{epoch:02d}-{val_loss:2f}',
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.monitor_loss,
        prefix='')

    tb_logger = loggers.TensorBoardLogger(save_dir=hparams.logger_save_dir,
                                          name=hparams.experiment_name)
    logger_list = [tb_logger]
    if hparams.checkpoint_path is None:
        model = vnet.VNet(hparams)
    else:
        model = vnet.VNet.load_from_checkpoint(hparams.checkpoint_path)

    trainer = Trainer.from_argparse_args(
        hparams,
        callbacks=[checkpoint_callback],
        logger=logger_list)

    trainer.fit(model)


if __name__ == '__main__':
    now = datetime.datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    parser = ArgumentParser()
    parser.add_argument('--logger_save_dir', default='D:/tmp/logs/november/')
    parser.add_argument('--monitor_loss', default='val_loss')
    parser.add_argument('--save_top_k', default=1, type=int)
    parser.add_argument('--experiment_name', default='vnet_testing_sparse')
    parser.add_argument('--date_time', default=dt_string)
    parser.add_argument('--checkpoint_path', default=None)

    parser = vnet.VNet.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # Override pytorch_lightning defaults
    parser.set_defaults(max_epochs=5000, gpus=1)

    hparams = parser.parse_args()

    main(hparams)
