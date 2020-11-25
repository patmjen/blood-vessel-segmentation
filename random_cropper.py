import torch
import rising.transforms.functional as F
from rising.transforms.abstract import BaseTransformSeeded
from rising.transforms.functional.crop import random_crop

def batch_random_crop(data, bs, size, dist = 0):
    """"
    Crop random patch/volume from input tensor

    Args:
        data: input tensor
        bs: batch size i.e. number of crops to make
        size: size of patch/volume
        dist: minimum distance to border. By default zero

    Returns:
        torch.Tensor: cropped outputs
        List[List[int]]: top left corners used for crops
    """
    return torch.cat([random_crop(data, size, dist) for _ in range(bs)])


class BatchRandomCrop(BaseTransformSeeded):
    def __init__(self, size, bs, dist = 0, keys = ('data',), grad = False, **kwargs):
        """
        Args:
            size: size of crop
            bs: batch size i.e. number of crops to make
            dist: minimum distance to border. By default zero
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=batch_random_crop,
                         keys=keys, size=size, bs=bs, dist=dist, grad=grad,
                         property_names=('size', 'dist'), **kwargs)