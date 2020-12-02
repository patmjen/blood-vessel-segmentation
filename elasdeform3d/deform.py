from .rbf import Rbf
import torch.nn.functional as F
import torch


def elastic_deform_3d(input, grid_spacing, std, interp_mode='nearest'):
    """
    Random elastic 3D deformation as described in Çiçek et al. [C16].

    Defines a set of displacement vectors on a coarse grid and then uses thin-
    plate spline interpolation to get dense deformation field. Displacement
    vectors are sampled from a zero mean Gaussian distribution with standard
    deviation `std`. Distance between points on the coarse grid is given by
    `grid_spacing`. Displacements at the input borders are set to zero.

    Parameters
    ----------
    input
        Input of shape N x C x D x H x W. Must be convertible to torch.Tensor.
    grid_spacing
        Distance between sampled displacement vectors.
    std
        Standard deviation of Gaussian distribution for displacement vectors.
    interp_mode
        Interpolation mode. Must be 'nearest' (default) or 'linear'.

    Returns
    -------
    torch.Tensor
        Deformed output of shape N x C x D x H x W.

    References
    ----------
    [C16] Çiçek et al., "3D U-Net: Learning Dense Volumetric Segmentation from
          Sparse Annotation", MICCAI. (pp 424-432). 2016.
    """
    # Validate inputs
    if input.ndim != 5:
        raise ValueError("Input must have shape N x C x D x H x W")

    if grid_spacing <= 0:
        raise ValueError("grid_spacing must be positive")

    if std < 0:
        raise ValueError("std must be non-negative")

    if interp_mode not in ['nearest', 'linear']:
        raise ValueError("interp_mode must be 'nearest' or 'linear'")
    if interp_mode == 'linear':
        interp_mode = 'bilinear'  # Change to fit with grid_sample naming

    input = torch.as_tensor(input)
    device = input.device
    vol_size = input.size()[-3:]
    disp_grid = torch.meshgrid(
        torch.arange(0, vol_size[0] + 1, grid_spacing, device=device),
        torch.arange(0, vol_size[1] + 1, grid_spacing, device=device),
        torch.arange(0, vol_size[2] + 1, grid_spacing, device=device))

    output = torch.empty_like(input)

    domain_grid = torch.meshgrid(
        torch.arange(vol_size[0], dtype=torch.float32, device=device),
        torch.arange(vol_size[1], dtype=torch.float32, device=device),
        torch.arange(vol_size[2], dtype=torch.float32, device=device))

    disp_vectors = torch.randn(*disp_grid[0].shape, 3,
                                dtype=torch.float32, device=device)
    disp_vectors *= std
    # Set outermost displacements to zero so interpolated vectors will
    # (mostly) stay within the volume.
    disp_vectors[0,:,:,:] = 0
    disp_vectors[-1,:,:,:] = 0
    disp_vectors[:,0,:,:] = 0
    disp_vectors[:,-1,:,:] = 0
    disp_vectors[:,:,0,:] = 0
    disp_vectors[:,:,-1,:] = 0

    # Save thin plate sline interpolator (similar to B-spline)
    spline = Rbf(*disp_grid, disp_vectors.reshape(-1, 3),
                    function='thin_plate', mode='N-D')
    disp = spline(*domain_grid)

    sample_grid = torch.stack(domain_grid, dim=3) + disp
    # Normalize grid coordinates
    sample_grid = sample_grid / torch.as_tensor(vol_size, device=device)
    sample_grid = 2 * sample_grid - 1  # Coords. must be in [-1, 1]
    sample_grid = sample_grid.expand(input.shape[0], -1, -1, -1, -1)
    output = F.grid_sample(input, sample_grid, mode=interp_mode,
                           padding_mode='border')

    return output.reshape(input.size())
