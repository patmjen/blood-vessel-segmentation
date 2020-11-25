from rising.loading import Dataset
import rising.transforms.functional as F
from os import listdir
from os.path import isfile, join
import torch
import numpy as np


class SubvolsDataset(Dataset):
    def __init__(self, data_dir, get_crop_fn):
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

        # Load all volumes
        self.volumes = [torch.stack(self.load_volume(i))
                        for i in range(self.num_volumes)]


    def get_crop(self, mask_and_data, index):
        return self.get_crop_fn(mask_and_data, index)


    def get_sample(self, index):
        data_and_mask = self.get_crop(index)
        return {'data': data_and_mask[0], 'label': data_and_mask[1] }


    def __getitem__(self, index: int) -> dict:
        """
        Gets a single sample

        Args:
            index: index specifying which item to load

        Returns:
            dict: the loaded sample
        """
        return self.get_sample(index)


    def load_volume(self, index: int) -> tuple:
        """
        Loads a single volume

        Args:
            index: index specifying which item to load

        Returns:
            np.array: loaded sample
            np.array: loaded label mask
        """
        file_id = self.file_ids[index]

        data = np.load(self.data_dir + 'data_' + file_id + '.npy')
        data = data[np.newaxis, ...]
        data = torch.from_numpy(data).float()

        mask = np.load(self.data_dir + 'mask_' + file_id + '.npy')
        mask = mask[np.newaxis, ...]
        mask = torch.from_numpy(mask).float()

        return data, mask


class RandomSubvolsDataset(SubvolsDataset):
    def __init__(self, data_dir, size, dist=0, samples_per_volume=10):
        super().__init__(data_dir, self.get_crop)
        self.size = size
        self.dist = dist
        self.samples_per_volume = samples_per_volume


    def get_crop(self, index):
        vol_index = index // self.samples_per_volume
        data_and_mask = self.volumes[vol_index]
        return F.random_crop(data_and_mask, self.size, self.dist)


    def __len__(self):
        """
        Number of samples per epoch

        Returns:
            int: samples per epoch
        """
        return self.num_volumes * self.samples_per_volume


class RandomSupportedSubvolsDataset(SubvolsDataset):
    def __init__(self, data_dir, size, dist=0, samples_per_volume=10):
        super().__init__(data_dir, self.get_crop)
        self.size = size
        self.dist = dist
        self.samples_per_volume = samples_per_volume


    def get_crop(self, index):
        vol_index = index // self.samples_per_volume
        data_and_mask = self.volumes[vol_index]
        # For now, just keep sampling random subvolumes until we find one with
        # labels. Since F.random_crop is fast, this is okay.
        while True:
            sample = F.random_crop(data_and_mask, self.size, self.dist)
            if torch.any(sample[1] > 0):
                return sample


    def __len__(self):
        """
        Number of samples per epoch

        Returns:
            int: samples per epoch
        """
        return self.num_volumes * self.samples_per_volume


class AllSubvolsDataset(SubvolsDataset):
    def __init__(self, data_dir, size):
        super().__init__(data_dir, self.get_crop)
        self.size = np.array(size)

        # Compute number of subvolumes needed to cover a volume.
        # NOTE: this assumes all volumes have the same size.
        vol_size = np.array(self.volumes[0].size()[-3:])
        assert np.all(size <= vol_size)
        self.samples_per_dim = vol_size // size + (vol_size % size > 0)
        self.samples_per_volume = self.samples_per_dim.prod()
        self.vol_size = vol_size


    def get_crop(self, index):
        vol_index = index // self.samples_per_volume
        sample_index = index % self.samples_per_volume

        corner = np.unravel_index(sample_index, self.samples_per_dim)
        corner = np.array(corner) * self.size
        # If a subvolume would exit the volume we move the corner back. This
        # means two subvolumes may overlap. The alternative would be padding.
        corner = np.minimum(self.vol_size - self.size, corner)

        data_and_mask = self.volumes[vol_index]
        return F.crop(data_and_mask, corner, self.size)


    def __len__(self):
        """
        Number of samples per epoch

        Returns:
            int: samples per epoch
        """
        return self.num_volumes * self.samples_per_volume


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

        data = np.load(self.data_dir + 'data_' + file_id + '.npy')
        data = data[np.newaxis, ...]
        data = torch.from_numpy(data).float()

        mask = np.load(self.data_dir + 'mask_' + file_id + '.npy')
        mask = mask[np.newaxis, ...]
        mask = torch.from_numpy(mask).float()

        return data, mask
