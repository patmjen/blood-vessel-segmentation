import os
from pytorch_lightning import Trainer
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
import vnet
import datetime
import pytorch_lightning as pl


def main(hparams):
    checkpoint_callback = ModelCheckpoint(
        filepath=hparams.logger_save_dir+hparams.experiment_name+'/ckpts/',
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.monitor_loss,
        prefix=''
    )

    tb_logger = loggers.TensorBoardLogger(save_dir=hparams.logger_save_dir, name=hparams.experiment_name)
    logger_list = [tb_logger]
    if hparams.checkpoint_path is None:
        model = vnet.VNet(hparams)
        trainer = Trainer(gpus=hparams.gpus, logger=logger_list, max_epochs=hparams.max_epochs,
                      checkpoint_callback=checkpoint_callback,
                      accumulate_grad_batches=hparams.accumulate_grad_batches,
                      replace_sampler_ddp=False)
    else:
        model = vnet.VNet.load_from_checkpoint(hparams.checkpoint_path)
        trainer = Trainer(gpus=hparams.gpus, logger=logger_list, max_epochs=hparams.max_epochs,
                      checkpoint_callback=checkpoint_callback,
                      accumulate_grad_batches=hparams.accumulate_grad_batches,
                      replace_sampler_ddp=False)

    trainer.fit(model)


if __name__ == '__main__':
    now = datetime.datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    parser = ArgumentParser()
    parser.add_argument('--lr', default=1e-3)
    parser.add_argument('--data_dir', default=os.getcwd() + '/data/')
    parser.add_argument('--train_loss_function', default="DiceLoss")
    parser.add_argument('--val_loss_function', default="DiceLoss")
    parser.add_argument('--max_epochs', default=5000, type=int)
    parser.add_argument('--logger_save_dir', default='D:/tmp/logs/november/')
    parser.add_argument('--monitor_loss', default='val_loss')
    parser.add_argument('--save_top_k', default=1, type=int)
    parser.add_argument('--experiment_name', default='vnet_testing')
    parser.add_argument('--date_time', default=dt_string)
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--accumulate_grad_batches', default=1, type=int)
    parser.add_argument('--checkpoint_path', default=None)
    parser.add_argument('--num_loader_workers', default=0, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--crop_size', default=128, type=int)
    parser.add_argument('--samples_per_volume', default=10, type=int)
    hparams = parser.parse_args()

    hparams.crop_size = (hparams.crop_size,) * 3

    main(hparams)
