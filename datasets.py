from os import listdir
from os.path import isfile, join

import torch
import numpy as np
from rising.loading import Dataset
import rising.transforms.functional as F
from tqdm import tqdm

class SubvolsDataset(Dataset):
    def __init__(self, data_dir, get_crop_fn, pre_load=True):
        self.data_dir = data_dir

        files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

        file_ids = []
        data_ids = []

        for filename in files:
            typ = filename.split('_', -1)[0]
            if typ == 'data':
                idx = filename.split('_', 1)[1]
                idx = idx.split('.', 1)[0]
                data_ids.append(idx)
                file_ids.append(idx)

        print('Data ids: ', data_ids)
        self.file_ids = file_ids
        self.num_volumes = len(file_ids)
        self.get_crop_fn = get_crop_fn

        self.pre_load = pre_load
        if self.pre_load:
            # Load all volumes
            self.volumes = [self.load_volume(i)
                            for i in range(self.num_volumes)]


    def get_sample(self, index):
        data_and_mask = self.get_crop_fn(index)
        return {'data': data_and_mask[0], 'label': data_and_mask[1] }


    def __getitem__(self, index: int) -> dict:
        """
        Gets a single sample

        Args:
            index: index specifying which sample to get.

        Returns:
            dict: the loaded sample
        """
        return self.get_sample(index)


    def get_volume(self, index: int) -> tuple:
        """
        Get or load a single volume.

        Args:
            index: index specifying which volume to get.

        Returns:
            np.array: loaded sample
            np.array: loaded label mask
        """
        if self.pre_load:
            return self.volumes[index]
        else:
            return self.load_volume(index)


    def load_volume(self, index: int) -> tuple:
        """
        Loads a single volume.

        Args:
            index: index specifying which volume to load.

        Returns:
            np.array: loaded sample
            np.array: loaded label mask
        """
        file_id = self.file_ids[index]

        data = np.load(join(self.data_dir, 'data_' + file_id + '.npy'))
        data = data[np.newaxis, ...]
        data = torch.from_numpy(data).float()

        mask = np.load(join(self.data_dir, 'mask_' + file_id + '.npy'))
        mask = mask[np.newaxis, ...]
        mask = torch.from_numpy(mask)

        return torch.stack((data, mask))


class RandomSubvolsDataset(SubvolsDataset):
    def __init__(self, data_dir, size, dist=0, samples_per_volume=10,
                 pre_load=True):
        super().__init__(data_dir, self.get_crop, pre_load=pre_load)
        self.size = size
        self.dist = dist
        self.samples_per_volume = samples_per_volume


    def get_crop(self, index):
        vol_index = index // self.samples_per_volume
        data_and_mask = self.get_volume(vol_index)
        return F.random_crop(data_and_mask, self.size, self.dist)


    def __len__(self):
        """
        Number of samples per epoch

        Returns:
            int: samples per epoch
        """
        return self.num_volumes * self.samples_per_volume


class RandomSupportedSubvolsDataset(SubvolsDataset):
    def __init__(self, data_dir, size, dist=0, samples_per_volume=10,
                 pre_load=True):
        super().__init__(data_dir, self.get_crop, pre_load=pre_load)
        self.size = size
        self.dist = dist
        self.samples_per_volume = samples_per_volume


    def get_crop(self, index):
        vol_index = index // self.samples_per_volume
        data_and_mask = self.get_volume(vol_index)
        # For now, just keep sampling random subvolumes until we find one with
        # labels. Since F.random_crop is fast, this is okay.
        while True:
            sample, corner = F.random_crop(data_and_mask, self.size, self.dist,
                                           return_corner=True)
            if torch.any(sample[1] > 0):
                corner = self._move_to_center_of_mass(corner, sample[1],
                                                      data_and_mask[1].size())
                return F.crop(data_and_mask, corner, self.size)


    def _move_to_center_of_mass(self, corner, labels, vol_size):
        x, y, z = torch.nonzero(labels > 0, as_tuple=True)[-3:]
        x = torch.round(x.float().mean() - self.size[0] / 2) + corner[0]
        y = torch.round(y.float().mean() - self.size[1] / 2) + corner[1]
        z = torch.round(z.float().mean() - self.size[2] / 2) + corner[2]
        x = min(max(x, 0), vol_size[-3] - self.size[0])
        y = min(max(y, 0), vol_size[-2] - self.size[1])
        z = min(max(z, 0), vol_size[-1] - self.size[2])
        return torch.tensor([x, y, z]).long()


    def __len__(self):
        """
        Number of samples per epoch

        Returns:
            int: samples per epoch
        """
        return self.num_volumes * self.samples_per_volume


class SubvolCorners:
    """
    Compute corner positions for moving over a volume with subvolumes of a
    given size and step.
    """
    def __init__(self, vol_size, size, step=None, border=None):
        """
        Args:
            vol_size: size of volume.
            size: size of subvolume.
            step: step length. Default is step=size.
            border: crops an extra part around each subvol, e.g. to get over-
                    lapping subvols. Default is no border.
        """
        self.vol_size = np.asarray(vol_size)
        self.size = np.asarray(size)
        if border is None:
            self.border = border
        else:
            self.border = np.asarray(border)

        if step is None:
            self.step = self.size
        else:
            self.step = np.asarray(step)

        self.samples_per_dim = vol_size // self.step \
                             + (vol_size % self.step > 0)
        self.total_samples = self.samples_per_dim.prod()


    def __getitem__(self, index):
        """
        Get corner position for i'th subvolume.

        Args:
            index: subvolume index.

        Returns:
            - If border is none:
              np.array: corner position
            - If border is not none
              np.array: outer corner position (including border)
              np.array: crop size including border
              np.array: inner corner position in cropped subvol
        """
        corner = np.unravel_index(index, self.samples_per_dim)
        corner = np.array(corner) * self.step
        # If a subvolume would exit the volume we move the corner back. This
        # means overlap may increase. The alternative would be padding.
        corner = np.minimum(self.vol_size - self.size, corner)
        if self.border is None:
            return corner
        else:
            outer_corner = np.maximum(0, corner - self.border)
            outer_size = np.minimum(self.vol_size - outer_corner,
                                    self.size + 2 * self.border)
            inner_corner = corner - outer_corner
            return outer_corner, outer_size, inner_corner


    def __iter__(self):
        """Iterator over corner positions."""
        for i in range(self.total_samples):
            yield self[i]


    def __len__(self):
        """Total number of subvolumes in volume."""
        return self.total_samples


class AllSubvolsDataset(SubvolsDataset):
    def __init__(self, data_dir, size, step=None, pre_load=True):
        super().__init__(data_dir, self.get_crop, pre_load=pre_load)
        # Compute number of subvolumes needed to cover a volume.
        # NOTE: this assumes all volumes have the same size.
        self.size = size
        vol_size = np.array(self.volumes[0].size()[-3:])
        assert np.all(size <= vol_size)
        self.subvol_corners = SubvolCorners(vol_size, size, step)
        self.samples_per_volume = len(self.subvol_corners)
        self.vol_size = vol_size


    def get_crop(self, index):
        vol_index = index // self.samples_per_volume
        sample_index = index % self.samples_per_volume

        corner = self.subvol_corners[sample_index]

        data_and_mask = self.get_volume(vol_index)
        return F.crop(data_and_mask, corner, self.size)


    def __len__(self):
        """
        Number of samples per epoch

        Returns:
            int: samples per epoch
        """
        return self.num_volumes * self.samples_per_volume


class AllSupportedSubvolsDataset(SubvolsDataset):
    def __init__(self, data_dir, size, pre_load=True):
        super().__init__(data_dir, self.get_crop, pre_load=pre_load)
        self.size = size
        self.vol_size = np.array(self.get_volume(0).size()[-3:])

        self.corner_lists = []
        self.index_to_vol = []
        print('Precomputing crops with labels ...')
        for i in range(self.num_volumes):
            corners = self._compute_supported_crop_corners(
                self.get_volume(i)[1])
            self.corner_lists.append(corners)
            self.index_to_vol += [i] * len(corners)

        corner_list_lengths = list(map(len, self.corner_lists))
        print('Number of crops:', corner_list_lengths)
        index_offsets = list(np.cumsum(corner_list_lengths))
        self.corner_index_offsets = [0] + index_offsets[:-1]


    def _compute_supported_crop_corners(self, labels):
        """
        Returns corners for all crops that contain labels
        """
        support = labels > 0
        corners = []
        for c in tqdm(SubvolCorners(self.vol_size, self.size)):
            if (F.crop(support, c, self.size) > 0).any():
                corners.append(c)
        return corners


    def get_crop(self, index):
        vol_index = self.index_to_vol[index]
        corner_index = index - self.corner_index_offsets[vol_index]
        corner = self.corner_lists[vol_index][corner_index]
        data_and_mask = self.get_volume(vol_index)
        return F.crop(data_and_mask, corner, self.size)


    def __len__(self):
        return len(self.index_to_vol)


class VnetDataset(Dataset):
    def __init__(self, data_dir: str, pre_load: bool = False):
        """
        Args:
            data_dir: directory containing the data
            pre_load: load all data into RAM upfront
        """

        self.data_dir = data_dir

        files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

        file_ids = []
        data_ids = []

        for filename in files:
            typ = filename.split('_', -1)[0]
            if typ == 'data':
                idx = filename.split('_', 1)[1]
                idx = idx.split('.', 1)[0]
                data_ids.append(idx)
                file_ids.append(idx)

        print('Data ids: ', data_ids)
        self.file_ids = file_ids
        self.num_samples = len(file_ids)

        self.pre_load = pre_load
        if pre_load:
            self.all_data = [self.load_sample(i)
                             for i in range(self.num_samples)]


    def __getitem__(self, index: int) -> dict:
        """
        Gets a single sample

        Args:
            index: index specifying which item to load

        Returns:
            dict: the loaded sample
        """
        return self.get_sample(index)


    def get_sample(self, index: int) -> dict:
        """
        Gets a single sample

        Args:
            index: index specifying which item to load

        Returns:
            dict: the loaded sample
        """
        if self.pre_load:
            data, mask = self.all_data[index]
        else:
            data, mask = self.load_sample(index)
        # return {'data': torch.from_numpy(data).float(),
        #         'label': torch.from_numpy(mask).float()}
        return { 'data': data, 'label': mask }


    def __len__(self) -> int:
        """
        Adds a length to the dataset

        Returns:
            int: dataset's length
        """
        return self.num_samples


    def load_sample(self, index: int) -> tuple:
        """
        Loads a single sample

        Args:
            index: index specifying which item to load

        Returns:
            np.array: loaded sample
            np.array: loaded label mask
        """
        file_id = self.file_ids[index]

        data = np.load(join(self.data_dir, 'data_' + file_id + '.npy'))
        data = data[np.newaxis, ...]
        data = torch.from_numpy(data).float()

        mask = np.load(join(self.data_dir, 'mask_' + file_id + '.npy'))
        mask = mask[np.newaxis, ...]
        mask = torch.from_numpy(mask).float()

        return data, mask
