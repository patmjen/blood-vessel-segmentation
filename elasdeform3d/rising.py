from rising.transforms.abstract import BaseTransformSeeded
from .deform import elastic_deform_3d


class ElasticDeformer3d(BaseTransformSeeded):
    """
    Simple wrapper class which exposes the elastic_deforma_3d as a seeded
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
            Interpolation mode or dict mapping keys to interpolations mode.
            Modes must be 'nearest' (default) or 'linear'.
        """
        super().__init__(augment_fn=self.deform, keys=keys)
        self.grid_spacing = grid_spacing
        self.std = std
        if isinstance(interp_mode, str):
            self.interp_modes = { k: interp_mode for k in keys }
        else:
            self.interp_modes = interp_mode


    def deform(self, data_and_key):
        data, key = data_and_key
        return elastic_deform_3d(data, self.grid_spacing, self.std,
            interp_mode=self.interp_modes[key])


    def forward(self, **data):
        """
        Apply transformation and use same seed for every key

        Parameters
        ----------
            data: dict with tensors

        Returns
        -------
            dict: dict with augmented data
        """
        # Add the key to each entry before sending it further
        return super().forward(**{ k: (d, k) for k, d in data.items() })
