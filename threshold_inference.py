import numpy as np
from os import listdir
from os.path import isfile, join

pred_dir = "/home/AG_Salditt/Messzeiten/2020/GINIX/run95_LTP/offline_analysis/ME/Rekos_fuerOve/temp_data/pred_val/"
gt_dir = "/home/AG_Salditt/Messzeiten/2020/GINIX/run95_LTP/offline_analysis/ME/Rekos_fuerOve/temp_data/gt_val/"

onlyfiles = [f for f in listdir(pred_dir) if isfile(join(pred_dir, f))]

size = (1060, 1000, 1000)


def binary_dice_coefficient(pred, gt, thresh, smooth: float = 1e-7):

    pred_flat = np.ndarray.flatten(pred)
    gt_flat = np.ndarray.flatten(gt)

    pred_bool = pred_flat > thresh

    intersec = (pred_bool * gt_flat).astype('float')
    return 2 * np.sum(intersec)/(np.sum(pred_bool).astype('float')+np.sum(gt_flat).astype('float')+smooth)


def dice_coefficient(pred, gt, thresh, smooth: float = 1e-7):

    pred_flat = np.ndarray.flatten(pred)
    gt_flat = np.ndarray.flatten(gt)

    pred_bool = pred_flat

    intersec = (pred_bool * gt_flat).astype('float')
    return 2 * np.sum(intersec)/(np.sum(pred_bool).astype('float')+np.sum(gt_flat).astype('float')+smooth)


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)  # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore:  # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)]  # mean accross images if per_image
    return 100 * np.array(ious)


pred_datas = []
gt_datas = []

for filename in onlyfiles:
    if filename.split('_', -1)[1] == 'mask':
        pred_data = np.reshape(np.fromfile(pred_dir+filename, dtype=np.float32), size)
        pred_datas.append(pred_data)
        gt_data = np.reshape(np.fromfile(gt_dir+filename, dtype=np.float32), size)
        gt_datas.append(gt_data)

idx = 0
highest_dice = 0
thresholds = np.arange(0.00, 1.00, 0.001)
dices = np.zeros((thresholds.shape[0]))
whole = np.zeros((2, thresholds.shape[0]))

for i, thresh in enumerate(thresholds):
    dice = 0
    for n in range(len(pred_datas)):
        dice += binary_dice_coefficient(pred_data[n], gt_data[n], thresh)
    dice = dice/len(pred_datas)
    dices[i] = dice

    if dice > highest_dice:
        highest_dice = dice
        idx = thresh

dice_coef = 0
for n in range(len(pred_datas)):
    dice_coef += dice_coefficient(pred_data[n], gt_data[n], thresh)
dice_coef = dice_coef/len(pred_datas)

print('Pred data len: ', len(pred_datas))
print('Dice coeff: ', dice_coef)
print('Best dice: ', highest_dice, 'Best thresh: ', idx)
whole[0, :] = thresholds
whole[1, :] = dices

np.save(pred_dir+"3215_val_inference", whole)
