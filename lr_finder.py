import os
import vnet
from pytorch_lightning import Trainer
from argparse import ArgumentParser
import datetime


def main(hparams):

    if hparams.checkpoint_path is None:
        model = vnet.VNet(hparams)
        trainer = Trainer(gpus=hparams.gpus, accumulate_grad_batches=hparams.accumulate_grad_batches)
    else:
        model = vnet.VNet.load_from_checkpoint(hparams.checkpoint_path)
        trainer = Trainer(gpus=hparams.gpus, accumulate_grad_batches=hparams.accumulate_grad_batches)

    lr_finder = trainer.lr_find(model, min_lr=hparams.min_lr, max_lr=hparams.max_lr)
    suggested_lr = lr_finder.suggestion()
    print("Suggested learning rate: ", suggested_lr)


if __name__ == '__main__':
    now = datetime.datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    parser = ArgumentParser()
    parser.add_argument('--lr', default=1e-5)
    parser.add_argument('--min_lr', default=1e-9)
    parser.add_argument('--max_lr', default=1)
    parser.add_argument('--data_dir', default='/home/ohansen/Documents/data/new_data/npy_512_noisy_student/')
    parser.add_argument('--train_loss_function', default="DiceLoss")
    parser.add_argument('--val_loss_function', default="DiceLoss")
    parser.add_argument('--max_epochs', default=5000)
    parser.add_argument('--logger_save_dir', default=os.getcwd()+'/logs/november/')
    parser.add_argument('--monitor_loss', default='val_loss')
    parser.add_argument('--save_top_k', default=3)
    parser.add_argument('--experiment_name', default='vnet_512_valsub2_diceloss_rc_allgood_datasets_from_ckpt_1970_1012_2nd_try')
    parser.add_argument('--date_time', default=dt_string)
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--accumulate_grad_batches', default=2)
    parser.add_argument('--checkpoint_path', default=None)
    hparams = parser.parse_args()

    main(hparams)
