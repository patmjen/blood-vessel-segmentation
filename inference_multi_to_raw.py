import vnet
from argparse import ArgumentParser
import torch
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.ndimage import zoom


def main(hparams):

    checkpoints = [
        '/home/ohansen/Documents/code/logs/september/vnet_512_valsub2_diceloss_rc_8datasets_from_ckpt_485/_ckpt_epoch_1970.ckpt'
    ]

    data_dir = '/home/AG_Salditt/Messzeiten/2020/GINIX/run96_LTP/offline_analysis/OH/npDataAndPrediction/'
    save_dir = '/home/AG_Salditt/Messzeiten/2020/GINIX/run96_LTP/offline_analysis/OH/NN_prediction/'

    name_addition = '_type=single_size=1000x1000x1060.raw'
    #orig_size = (512, 512, 512)
    #save_size = (1060, 1000, 1000)

    zoom_factors = (1060/512, 1000/512, 1000/512)

    # print('Zoom factors: ', zoom_factors)

    size = 512
    step_size = 256
    step = int(size/step_size)

    files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    for i, filename in enumerate(files):
        typ = filename.split('_', -1)[0]
        idx = filename.split('_', -1)[1]
        idx = idx.split('.', -1)[0]

        if typ == 'Covid':
            print('Current file: ', filename)

            data = np.load(data_dir + filename)
            reco_mask = np.zeros((size, size, size), dtype=np.float32)

            data = np.absolute(1-data)
            mean = 317.3661
            std = 2.83739
            inference_std = np.std(data)
            inference_mean = np.mean(data)
            data += (-inference_mean)
            data *= (std/inference_std)
            data += mean

            print('Data mean: ', np.mean(data))
            print('Data std: ', np.std(data))

            if idx == '4115':
                zoom_factors = (1004/512, 1000/512, 1000/512)
                name_addition = '_type=single_size=1000x1000x1004.raw'
            else:
                zoom_factors = (1060/512, 1000/512, 1000/512)
                name_addition = '_type=single_size=1000x1000x1060.raw'

            for i, ckpt in enumerate(checkpoints):
                print('Current model ckpt: ', ckpt)

                model = vnet.VNet.load_from_checkpoint(checkpoint_path=ckpt)
                device = torch.device(dev)
                model = model.to(device)
                model.eval()

                with torch.no_grad():
                    for x in range(step):
                        for y in range(step):
                            for z in range(step):

                                sub_data = data[x*step_size:(x+1)*step_size, y*step_size:(y+1)*step_size, z*step_size:(z+1)*step_size]
                                sub_data = sub_data[np.newaxis, np.newaxis, ...]
                                pred_mask = model(torch.from_numpy(sub_data).float().to(device))
                                # print('Pred mask shape: ', pred_mask.size(), ' x,y,z: ', x, y, z)
                                sig_pred_mask = torch.sigmoid(pred_mask.cpu()).detach().numpy()
                                sig_pred_mask = np.squeeze(sig_pred_mask)
                                reco_mask[x*step_size:(x+1)*step_size, y*step_size:(y+1)*step_size, z*step_size:(z+1)*step_size] += sig_pred_mask

            reco_mask = reco_mask / float(len(checkpoints))
            reco_mask = np.array((reco_mask > 0.123), dtype=np.float32)
            # reco_mask = zoom(reco_mask, zoom_factors, order=0)
            # print('Zoom factors: ', zoom_factors)
            # reco_mask = zoom(reco_mask, zoom_factors)
            reco_mask = reco_mask.astype(np.float32)
            np.save(save_dir+'prob_mask_512', reco_mask)
            reco_mask.tofile(save_dir+'prob_mask_512.raw')#+idx+name_addition)

        else:
            continue


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()

    main(args)
