import os
from os.path import join
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import rising.transforms as rtr
from rising.loading import DataLoader
from rising.transforms import Compose

import losses
import datasets
from init_weights import init_weights
from elasdeform3d.rising import ElasticDeformer3d


class ConvStep(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('Conv', nn.Conv3d(in_channels, out_channels,
                                          kernel_size=3, padding=1, 
                                          bias=False))
        self.add_module('BatchNorm', nn.BatchNorm3d(out_channels))
        self.add_module('ReLU', nn.ReLU())


class InputBlock(nn.Sequential):
    def __init__(self, in_channels, base_channels, out_channels=None, 
                 n_conv=1):
        super().__init__()
        if out_channels is None:
            out_channels = 2 * base_channels
        self.add_module('ConvStep0', ConvStep(in_channels, base_channels))
        for i in range(n_conv - 1):
            self.add_module(f'ConvStep{i + 1}',
                            ConvStep(base_channels, base_channels))
        self.add_module(f'ConvStep{n_conv}',
                        ConvStep(base_channels, out_channels))


class EncodeBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels=None, n_conv=2):
        super().__init__()
        assert(n_conv >= 1)
        if out_channels is None:
            out_channels = 2 * in_channels
        self.add_module('MaxPool', nn.MaxPool3d(2))
        for i in range(n_conv - 1):
            self.add_module(f'ConvStep{i}',
                            ConvStep(in_channels, in_channels))

        self.add_module(f'ConvStep{n_conv}',
                        ConvStep(in_channels, out_channels))


class Upsampler(nn.Module):
    def __init__(self, interp_mode='nearest'):
        super().__init__()
        self.interp_mode = interp_mode


    def forward(self, x, skip_x):
        return F.interpolate(x, skip_x.shape[2:], mode=self.interp_mode)


class DecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, skip_channels=None,
                 n_conv=2, interp_mode='nearest'):
        super().__init__()
        assert(n_conv >= 1)
        if out_channels is None:
            out_channels = in_channels // 2
        if skip_channels is None:
            skip_channels = out_channels
        self.upsampler = Upsampler(interp_mode)
        self.conv_steps = nn.Sequential()
        self.conv_steps.add_module(
            'ConvStep0', ConvStep(in_channels + skip_channels, out_channels))
        for i in range(n_conv - 1):
            self.conv_steps.add_module(f'ConvStep{i + 1}',
                                       ConvStep(out_channels, out_channels))


    def forward(self, x, skip_x):
        up_x = self.upsampler(x, skip_x)
        catx = torch.cat([up_x, skip_x], dim=1)
        return self.conv_steps(catx)


class UNet3d(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=32,
                 num_levels=4):
        super().__init__()
        self.input_conv = InputBlock(in_channels, base_channels)
        self.encoders = nn.ModuleList()
        for i in range(1, num_levels):
            self.encoders.append(EncodeBlock(base_channels * (2 ** i)))
        self.decoders = nn.ModuleList()
        for i in reversed(range(1, num_levels)):
            self.decoders.append(DecodeBlock(base_channels * (2 ** (i + 1))))
        self.output_conv = nn.Conv3d(2 * base_channels, out_channels,
                                     kernel_size=1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.GroupNorm):
                init_weights(m, init_type='kaiming')


    def forward(self, x):
        x_crnt = self.input_conv(x)
        skips = [x_crnt]
        for enc in self.encoders:
            x_crnt = enc(x_crnt)
            skips.append(x_crnt)
        skips.pop()
        for dec in self.decoders:
            x_skip = skips.pop()
            x_crnt = dec(x_crnt, x_skip)
        return F.softmax(self.output_conv(x_crnt), dim=1)


class UNet3dTrainer(pl.LightningModule):
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
        super().__init__()

        if not hasattr(hparams['crop_size'], '__len__'):
            hparams['crop_size'] = (hparams['crop_size'],) * 3

        self.save_hyperparameters(hparams)
        self.model = UNet3d(1, 2)


    def forward(self, x):
        return self.model(x)


    def training_step(self, train_batch, batch_idx):
        x, y = train_batch['data'], train_batch['label']
        pred = self.forward(x)

        loss = losses.SparseDiceLoss()

        # Label 1 is background and label 2 is vessel
        pred_0, pred_1 = pred.split(1, dim=1)
        res = 0.5 * (loss(pred_0, y, 2) + loss(pred_1, y, 1))

        self.log('train_loss', res, on_step=True, prog_bar=True, logger=True)
        return res


    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch['data'], val_batch['label']
        pred = self.forward(x)

        loss = losses.SparseDiceLoss()

        # Label 1 is background and label 2 is vessel
        pred_0, pred_1 = pred.split(1, dim=1)
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
        self.train_dataset = datasets.RandomSupportedSubvolsDataset(
            data_dir=join(self.hparams.data_dir, 'train'),
            size=self.hparams.crop_size,
            samples_per_volume=self.hparams.samples_per_volume)

        self.val_dataset = datasets.AllSupportedSubvolsDataset(
            data_dir=join(self.hparams.data_dir, 'val/'),
            size=self.hparams.crop_size)


    def train_dataloader(self):
        keys = ('data', 'label')
        transforms_augment = []
        transforms_augment.append(rtr.Rot90(dims=(0, 1, 2), keys=keys))
        transforms_augment.append(rtr.Mirror(dims=(0, 1, 2), keys=keys))
        transforms_augment.append(ElasticDeformer3d(32, 4, keys=keys,
            interp_mode={ 'data': 'linear', 'label': 'nearest' }))
        gpu_transforms = Compose(transforms_augment)
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_loader_workers,
                          shuffle=True,
                          gpu_transforms=gpu_transforms,
                          pin_memory=True)


    def val_dataloader(self):
        gpu_transforms = []
        keys = ('data', 'label')
        gpu_transforms.append(rtr.Rot90(dims=(0, 1, 2), keys=keys))
        gpu_transforms.append(rtr.Mirror(dims=(0, 1, 2), keys=keys))
        gpu_transforms = Compose(gpu_transforms)

        return DataLoader(self.val_dataset,
                          batch_size=2 * self.hparams.batch_size,
                          num_workers=self.hparams.num_loader_workers,
                          shuffle=False,
                          gpu_transforms=gpu_transforms,
                          pin_memory=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, min_lr=1e-5, factor=0.8)
        return { 'optimizer': optimizer, 'lr_scheduler': scheduler,
                 'monitor': 'val_loss' }
