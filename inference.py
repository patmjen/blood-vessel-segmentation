import vnet
from argparse import ArgumentParser
import torch
import losses
import numpy as np
from os import listdir
from os.path import isfile, join


def main(hparams):

    model = vnet.VNet.load_from_checkpoint(
        checkpoint_path='/home/ohansen/Documents/code/logs/september/vnet_512_sub2_diceloss_rc_custvol_3_run/_ckpt_epoch_8.ckpt')

    data_dir = '/home/ohansen/Documents/data/inference/npy_512_all/'
    save_dir = '/home/ohansen/Documents/data/inference/inference_npy_512_all/'

    files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

    reco_data = np.zeros((512, 512, 512))
    true_mask = np.zeros((512, 512, 512))
    reco_mask = np.zeros((512, 512, 512))
    step = 256

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    device = torch.device(dev)

    model = model.to(device)

    model.eval()

    with torch.no_grad():
        for i, filename in enumerate(files):
            typ = filename.split('_', -1)[0]
            idx = filename.split('_', -1)[1]

            if typ == 'data':
                x = filename.split('_', -1)[2]
                x_int = int(x)
                y = filename.split('_', -1)[3]
                y_int = int(y)
                z = filename.split('_', -1)[4]
                z = z.split('.', -1)[0]
                z_int = int(z)

                print(x, y, z)

                sub_data = np.load(data_dir + filename)
                sub_mask = np.load(data_dir+'mask'+'_'+idx+'_'+x+'_'+y+'_'+z+'.npy')
                reco_data[x_int*step:(x_int+1)*step, y_int*step:(y_int+1)*step, z_int*step:(z_int+1)*step] = sub_data
                true_mask[x_int*step:(x_int+1)*step, y_int*step:(y_int+1)*step, z_int*step:(z_int+1)*step] = sub_mask
                sub_data = sub_data[np.newaxis, np.newaxis, ...]
                sub_mask = sub_mask[np.newaxis, np.newaxis, ...]
                print('Loaded data successfully')
                pred_mask = model(torch.from_numpy(sub_data).float().to(device))
                print('Pred mask shape: ', pred_mask.size())
                dice_coeff = losses.binary_dice_coefficient(pred_mask, torch.from_numpy(sub_mask).float().to(device))
                print('Binary Dice coefficient for subvolume ', x, y, z, ' :', dice_coeff)

                sig_pred_mask = torch.sigmoid(pred_mask.cpu()).detach().numpy()
                reco_mask[x_int*step:(x_int+1)*step, y_int*step:(y_int+1)*step, z_int*step:(z_int+1)*step] = sig_pred_mask

    np.save(save_dir + 'data_7516.npy', reco_data)
    np.save(save_dir + 'mask_7516.npy', true_mask)
    np.save(save_dir + 'pred_mask_7516.npy', reco_mask)


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()

    main(args)
