from argparse import ArgumentParser
import os
from os import listdir
from os.path import isfile, join, basename
import warnings

import torch
import numpy as np
import rising.transforms.functional as F
from tqdm import tqdm

with warnings.catch_warnings():
    # Avoid warnings from tensorboard uing deprecated functions
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import vnet
    import unet

from datasets import SubvolCorners

def main(args):
    print('Loading model')
    if args.model == 'vnet':
        model = vnet.VNet.load_from_checkpoint(args.checkpoint)
    else:  # args.model == 'unet'
        model = unet.UNet3dTrainer.load_from_checkpoint(args.checkpoint)

    size = np.asarray(args.crop_size)

    files = []
    for f in listdir(args.data_dir):
        if isfile(join(args.data_dir, f)):
            files.append(f)

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    device = torch.device(dev)
    model = model.to(device)
    model.eval()

    for i, filename in enumerate(files):
        print('File', i + 1, ':', filename)
        data = torch.from_numpy(np.load(join(args.data_dir, filename)))
        vol_size = data.shape
        data = data.unsqueeze(0).unsqueeze(0)

        pred = torch.zeros((2,) + vol_size, dtype=torch.float32)
        mask = torch.zeros((1,) + vol_size, dtype=torch.bool)

        for outer_corner, outer_size, inner_corner in tqdm(SubvolCorners(
            vol_size, size, border=args.border)):
            sub_data = F.crop(data, outer_corner, outer_size)
            sub_pred = F.crop(pred, outer_corner + inner_corner, size)
            sub_mask = F.crop(mask, outer_corner + inner_corner, size)

            if torch.cuda.is_available():
                sub_data = sub_data.cuda()

            with torch.no_grad():
                res = model(sub_data).cpu()
                sub_pred[:] = F.crop(res, inner_corner, size)
            sub_mask[:] = sub_pred.argmax(dim=0) == 1

        pred = pred.squeeze()
        mask = mask.squeeze()

        out_filename = join(args.save_dir,
                            basename(args.checkpoint) + '.' + filename)
        if args.file_type == 'npy':
            np.save(out_filename + '.pred', pred.numpy())
            np.save(out_filename + '.mask', mask.numpy())
        elif args.file_type == 'raw':
            pred.numpy().tofile(out_filename + '.pred.raw')
            mask.numpy().tofile(out_filename + '.mask.raw')
        else:
            raise ValueError('Invalid file type: {}'.format(args.file_type))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--save_dir', default=os.getcwd())
    parser.add_argument('--data_dir', default=os.getcwd())
    parser.add_argument('--crop_size', default=128, type=int)
    parser.add_argument('--file_type', default='npy', choices=['npy', 'raw'])
    parser.add_argument('--model', default='vnet', choices=['vnet', 'unet'])
    parser.add_argument('--border', default=32, type=int)
    args = parser.parse_args()

    args.crop_size = (args.crop_size,) * 3

    main(args)
