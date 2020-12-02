from rising.transforms.abstract import BaseTransformSeeded
from .deform import elastic_deform_3d
from .deform import ElasticDeformer3d as BaseElasticDeformer3d


class ElasticDeformer3d(BaseTransformSeeded):
    """
    Simple wrapper class which exposes the ElasticDeformer3d class as a seeded
    rising transform.
    """

    def __init__(self, grid_spacing, std, interp_mode='nearest',
                 keys=('data',)):
        """
        Initialize transform.

        Parameters
        ----------
        grid_spacing
            Distance between sampled displacement vectors.
        std
            Standard deviation of Gaussian distribution for displacement
            vectors.
        interp_mode
            Interpolation mode. Must be 'nearest' (default) or 'linear'.
        """
        super().__init__(
            augment_fn=BaseElasticDeformer3d(grid_spacing, std, interp_mode),
            keys=keys)
