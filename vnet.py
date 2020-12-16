# Modified code from https://github.com/mattmacy/vnet.pytorch

import os
from os.path import join
import pytorch_lightning as pl
import torch
import torch.nn as nn
import datasets
import rising.transforms as rtr
from rising.random import UniformParameter
from rising.loading import DataLoader
from rising.transforms import Compose
from init_weights import init_weights
import losses
import torch_summary
from argparse import ArgumentParser
from elasdeform3d.rising import ElasticDeformer3d


def passthrough(x, **kwargs):
    return x


def get_num_groups(num_chans):
    #if num_chans / 32 >= 8:
    #    # Ideally, we want 32 groups with at least 8 channels each
    #    return 32
    #elif num_chans / 8 >= 2:
    #    # Else we want at least two groups with 8 channels each
    #    return num_chans // 8
    #else:
    #    # Else we just do two groups unless we have less than 4 channels
    #    return 2 if num_chans >= 4 else num_chans
    return max(num_chans, num_chans // 32)


class ConvStep(nn.Module):
    def __init__(self, num_chans, dropout, do_rate):
        super(ConvStep, self).__init__()
        self.conv_1 = nn.Conv3d(num_chans, num_chans, kernel_size=5, padding=2,
                                bias=False)
        self.batch_norm_1 = nn.BatchNorm3d(num_chans)
        self.prelu_1 = nn.PReLU(num_chans)
        self.do_1 = passthrough
        if dropout:
            self.do_1 = nn.Dropout3d(p=do_rate)

    def forward(self, x):
        out = self.do_1(self.prelu_1(self.batch_norm_1(self.conv_1(x))))
        return out


def make_nConvs(num_chans, num_convs, dropout, do_rate):
    layers = []
    for _ in range(num_convs):
        layers.append(ConvStep(num_chans, dropout, do_rate))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, out_cha):
        super(InputTransition, self).__init__()
        self.conv_1 = nn.Conv3d(1, out_cha, kernel_size=5, padding=2,
                                bias=False)
        self.batch_norm_1 = nn.BatchNorm3d(out_cha)
        self.prelu_1 = nn.PReLU(out_cha)

    def forward(self, x):
        out = self.prelu_1(self.batch_norm_1(self.conv_1(x)))
        # Right cat dimension?
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 1)
        out = torch.add(out, x16)

        return out


class DownTransition(nn.Module):
    def __init__(self, in_chans, nConvs, dropout=False, do_rate=0.5):
        super(DownTransition, self).__init__()
        out_chans = 2*in_chans
        self.down_conv = nn.Conv3d(in_chans, out_chans, kernel_size=2,
                                   stride=2, bias=False)
        self.batch_norm_1 = nn.BatchNorm3d(out_chans)
        self.prelu_1 = nn.PReLU(out_chans)
        self.do_1 = passthrough
        if dropout:
            self.do_1 = nn.Dropout3d(p=do_rate)
        self.n_convs = make_nConvs(out_chans, nConvs, dropout, do_rate)

    def forward(self, x):
        down = self.do_1(self.prelu_1(self.batch_norm_1(self.down_conv(x))))
        out = self.n_convs(down)
        out = torch.add(out, down)
        return out


class UpTransition(nn.Module):
    def __init__(self, in_chans, out_chans, nConvs, dropout=False, do_rate=0.5):
        super(UpTransition, self).__init__()
        # Should the input to the nConvs have 256 or 384 channels?
        # This also affects the recurrent connection leading to the end of the nConvs
        out_chans_half = out_chans // 2
        # out_chans_half = out_chans
        self.up_conv = nn.ConvTranspose3d(
            in_chans, out_chans_half, kernel_size=2, stride=2, bias=False)
        self.batch_norm_1 = nn.BatchNorm3d(out_chans_half)
        self.prelu_1 = nn.PReLU(out_chans_half)
        self.do_1 = passthrough
        if dropout:
            self.do_1 = nn.Dropout3d(p=do_rate)
        self.n_convs = make_nConvs(out_chans, nConvs, dropout, do_rate)

    def forward(self, x, skipx):
        out_upConv = self.do_1(self.prelu_1(self.batch_norm_1(self.up_conv(x))))
        # Correct cat dimension?
        xcat = torch.cat((out_upConv, skipx), 1)
        out = self.n_convs(xcat)
        out = torch.add(out, xcat)
        return out


class OutputTransition(nn.Module):
    def __init__(self, in_chans):
        super(OutputTransition, self).__init__()
        self.conv_1 = nn.Conv3d(in_chans, 2, kernel_size=1, bias=False)
        self.bn_1 = nn.BatchNorm3d(2)
        self.prelu_1 = nn.PReLU(2)
        # Input should be N x C x D x H x W and we want max over C dimension.
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.prelu_1(self.bn_1(self.conv_1(x))))


class VNet(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        cwd = os.getcwd()
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', default=1e-3, type=float)
        parser.add_argument('--num_loader_workers', default=0, type=int)
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--crop_size', default=128, type=int)
        parser.add_argument('--samples_per_volume', default=10, type=int)
        parser.add_argument('--data_dir', default=join(cwd, 'data', 'sparse'))
        return parser

    def __init__(self, **hparams):
        super(VNet, self).__init__()

        if not hasattr(hparams['crop_size'], '__len__'):
            hparams['crop_size'] = (hparams['crop_size'],) * 3

        self.save_hyperparameters(hparams)

        self.in_tr = InputTransition(16)
        self.down_tr32 = DownTransition(16, 2)
        self.down_tr64 = DownTransition(32, 3)
        self.down_tr128 = DownTransition(64, 3)
        self.down_tr256 = DownTransition(128, 3)
        self.up_tr256 = UpTransition(256, 256, 3)
        self.up_tr128 = UpTransition(256, 128, 2)
        self.up_tr64 = UpTransition(128, 64, 2)
        self.up_tr32 = UpTransition(64, 32, 2)
        self.out_tr = OutputTransition(32)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.GroupNorm):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        return self.out_tr(out)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch['data'], train_batch['label']
        pred = self.forward(x)

        loss = losses.SparseDiceLoss()

        pred_0, pred_1 = pred.split(1, dim=1)
        # Label 1 is background and label 2 is vessel
        res = 0.5 * (loss(pred_0, y, 2) + loss(pred_1, y, 1))

        self.log('train_loss', res, on_step=True, prog_bar=True, logger=True)
        return res

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch['data'], val_batch['label']
        pred = self.forward(x)

        loss = losses.SparseDiceLoss()

        pred_0, pred_1 = pred.split(1, dim=1)
        # Label 1 is background and label 2 is vessel

        res = 0.5 * (loss(pred_0, y, 2) + loss(pred_1, y, 1))

        # Log prediction images
        data_slice = x[:, :, :, :, x.shape[-1] // 2]
        pred_slice = pred_1[:, :, :, :, x.shape[-1] // 2].squeeze(dim=1)
        log_im = data_slice.repeat([1, 3, 1, 1])
        log_im[:, 0, :, :] += 0.5 * pred_slice
        log_im = log_im.clamp(0, 1)

        self.log('val_loss', res, prog_bar=True, logger=True)
        return res, log_im

    def validation_epoch_end(self, val_step_outputs):
        # Split results in separate lists
        # https://stackoverflow.com/a/19343/1814397
        val_step_outputs = list(zip(*val_step_outputs))
        self.log('val_loss', torch.stack(val_step_outputs[0]).mean(),
                 prog_bar=True, logger=True)
        log_im = torch.cat(val_step_outputs[1])
        self.logger.experiment.add_images(f'Predictions', log_im,
                                          global_step=self.global_step)


    def prepare_data(self):
        print("Preparing data ...")
        # self.train_dataset = datasest.VnetDataset(pre_load=True, data_dir=self.hparams.data_dir+'train/')
        self.train_dataset = datasets.RandomSupportedSubvolsDataset(
            data_dir=join(self.hparams.data_dir, 'train'),
            size=self.hparams.crop_size,
            samples_per_volume=self.hparams.samples_per_volume)

        # self.val_dataset = datasets.VnetDataset(pre_load=True, data_dir=self.hparams.data_dir+'val/')
        self.val_dataset = datasets.AllSupportedSubvolsDataset(
            data_dir=join(self.hparams.data_dir, 'val/'),
            size=self.hparams.crop_size)

    def train_dataloader(self):
        transforms_augment_cpu = []
        transforms_augment = []

        #transforms_augment_cpu.append(rtr.intensity.RandomAddValue(UniformParameter(-0.2, 0.2)))
        #cpu_transforms = Compose(transforms_augment_cpu)

        keys = ('data', 'label')
        # transforms_augment.append(rtr.GaussianNoise(0., 0.05))
        transforms_augment.append(rtr.Rot90(dims=(0, 1, 2), keys=keys))
        transforms_augment.append(rtr.Mirror(dims=(0, 1, 2), keys=keys))
        transforms_augment.append(ElasticDeformer3d(32, 4, keys=keys,
            interp_mode={ 'data': 'linear', 'label': 'nearest' }))
        #transforms_augment.append(rtr.BaseAffine(
        #    scale=UniformParameter(0.95, 1.05),
        #    rotation=UniformParameter(-45, 45), degree=True,
        #    translation=UniformParameter(-0.05, 0.05),
        #    keys=('data', 'label'),
        #    interpolation_mode='nearest'))
        gpu_transforms = Compose(transforms_augment)
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_loader_workers,
                          shuffle=True,
                          #batch_transforms=cpu_transforms,
                          gpu_transforms=gpu_transforms,
                          pin_memory=True)
        # , sample_transforms=transforms_augment)

    def val_dataloader(self):
        gpu_transforms = []
        gpu_transforms.append(rtr.Rot90(dims=(0, 1, 2), keys=('data', 'label')))
        gpu_transforms.append(rtr.Mirror(dims=(0, 1, 2), keys=('data', 'label')))
        gpu_transforms = Compose(gpu_transforms)

        # batch_transforms = []
        # batch_transforms.append(BatchRandomCrop(self.hparams.crop_size, bs=1, dist=0, keys=('data', 'label')))
        # batch_transforms = Compose(batch_transforms)

        return DataLoader(self.val_dataset,
                          batch_size=2 * self.hparams.batch_size,
                          num_workers=self.hparams.num_loader_workers,
                          shuffle=False,
                          # batch_transforms=batch_transforms,
                          gpu_transforms=gpu_transforms,
                          pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, min_lr=1e-4)
        return { 'optimizer': optimizer, 'lr_scheduler': scheduler,
                 'monitor': 'val_loss' }

