from torch.utils.data import DataLoader
import os
from pathlib import Path
from torch.utils.data import Dataset
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import torch


def list_directory(path):
    return sorted(os.listdir(path))


def list_file_pairs(part, root='data/training'):
    path = Path(root)
    files = list_directory(path / part)
    for i in range(1, len(files), 2):
        yield path / part / files[i - 1], path / part / files[i]


def validate_file_pair(pair):
    return pair[0].name[0:-5] == pair[1].name[0:-5]


class ImagePairDataset(Dataset):
    def __init__(self, pairs, label, transforms=None):
        self.pairs = list(pairs)
        self.transforms = transforms
        self.label = label

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path_a, path_b = self.pairs[idx]
        img_a, img_b = Image.open(path_a), Image.open(path_b)
        img_a, img_b = img_a.convert('RGB'), img_b.convert('RGB')
        if self.transforms:
            img_a, img_b = self.transforms(img_a), self.transforms(img_b)
        return (img_a, img_b), self.label, (str(path_a), str(path_b))


def preview_image_data(dataset, transforms=None, permute=False, base_idx=None):
    fig = plt.figure(figsize=(12, 17))
    base_idx = np.random.randint(0, len(dataset) - 9) if base_idx is None else base_idx
    for i in range(3):
        ax1 = fig.add_subplot(3, 2, i * 2 + 1, xticks=[], yticks=[])
        ax2 = fig.add_subplot(3, 2, i * 2 + 2, xticks=[], yticks=[])
        image = dataset[base_idx + i][0][0] if transforms is None else transforms(dataset[base_idx + i][0][0])
        image2 = dataset[base_idx + i][0][1] if transforms is None else transforms(dataset[base_idx + i][0][1])
        if permute:
            image = image.permute(1, 2, 0)
            image2 = image2.permute(1, 2, 0)
        ax1.imshow(image)
        ax2.imshow(image2)
    fig.tight_layout()


def list_same_diff_pairs(root='/home/daniel/data_dogs/training'):
    same, diff = list(list_file_pairs(part='same', root=root)), list(list_file_pairs(part='different', root=root))
    return same, diff


def create_dataset(root, transforms=None):
    same, diff = list_same_diff_pairs(root)
    return ImagePairDataset(same, 1, transforms=transforms) + ImagePairDataset(diff, 0, transforms=transforms)


def create_dataset_test(root='/home/daniel/data_dogs/testing', transforms=None):
    return create_dataset(root, transforms)


def create_dataset_train(root='/home/daniel/data_dogs/training', transforms=None):
    return create_dataset(root, transforms)


def create_dataloader_train(root='/home/daniel/data_dogs/training', batch_size=32, num_workers=4, transforms=None,
                            shuffle=True):
    return create_dataloader(root, batch_size, num_workers, transforms, shuffle)


def create_dataloader_test(root='/home/daniel/data_dogs/testing', batch_size=32, num_workers=4, transforms=None,
                           shuffle=False):
    return create_dataloader(root, batch_size, num_workers, transforms, shuffle)


def create_dataloader_validation(root='/home/daniel/data_dogs/validation', batch_size=32, num_workers=4,
                                 transforms=None, shuffle=False):
    return create_dataloader(root, batch_size, num_workers, transforms, shuffle)


def create_dataloader(root, batch_size, num_workers, transforms, shuffle, drop_last=False):
    dataset = create_dataset(root, transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)


class FeaturesDataset(Dataset):
    def __init__(self, data_path, bidirectional=False, with_file_paths=False):
        import itertools
        data = torch.load(data_path)
        data_a, data_b, labels, f_paths_a, f_paths_b = [], [], [], [], []
        for (feat_a, feat_b), y_batch, (file_p_a, file_p_b) in data:
            data_a.append(feat_a)
            data_b.append(feat_b)
            labels.append(y_batch)
            f_paths_a.append(file_p_a)
            f_paths_b.append(file_p_b)

        self.features_a = torch.cat(data_a)
        self.features_b = torch.cat(data_b)
        self.labels = torch.cat(labels)
        self.paths_a = list(itertools.chain(*f_paths_a))
        self.paths_b = list(itertools.chain(*f_paths_b))
        self.bidirectional = bidirectional
        self.with_file_paths = with_file_paths

        if self.features_a.size(0) != self.features_b.size(0) or self.features_a.size(0) != self.labels.size(0):
            raise ValueError('data and labels must have the same length')

    def __len__(self):
        return self.features_a.size(0) if not self.bidirectional else self.features_a.size(0) * 2

    def _get_item(self, idx):
        if self.with_file_paths:
            return (self.features_a[idx], self.features_b[idx]), self.labels[idx], (
                self.paths_a[idx], self.paths_b[idx])
        return (self.features_a[idx], self.features_b[idx]), self.labels[idx]

    def _get_item_rev(self, idx):
        if self.with_file_paths:
            return (self.features_b[idx], self.features_a[idx]), self.labels[idx], (
                self.paths_a[idx], self.paths_b[idx])
        return (self.features_b[idx], self.features_a[idx]), self.labels[idx]

    def __getitem__(self, idx):
        if not self.bidirectional:
            return self._get_item(idx)
        if idx % 2 == 0:
            i = idx // 2
            return self._get_item(i)
        else:
            i = idx // 2
            return self._get_item_rev(i)


from torch.utils.data import Dataset
import h5py
import torch

def init_cache():
    return {
        "features_a": None,
        "features_b": None,
        "labels": None,
        "paths_a": None,
        "paths_b": None,
        "current_cache_from": None,
        "current_cache_to": None
    }


class FeaturesDatasetHDF5(Dataset):
    def __init__(self, data_path, with_file_paths=False):
        self.worker_file_dict = {}
        self.worker_cache_dict = {}
        self.file_path = data_path
        self.with_file_paths = with_file_paths
        self.local_cache = init_cache()

        with h5py.File(self.file_path, "r") as hfd5_file:
            self.batch_size = hfd5_file["entry_0"]["features_a"].shape[0]
            self.num_keys = len(hfd5_file.keys())
            self.update_cache(hfd5_file, 0, self.local_cache)
            self.last_batch_size = hfd5_file[f"entry_{self.num_keys - 1}"]["features_a"].shape[0]

    def __len__(self):
        return self.num_keys * self.batch_size - (self.batch_size - self.last_batch_size)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError('index out of range')

        worker_info = torch.utils.data.get_worker_info()
        hfd5_file = None if worker_info is None else self.worker_file_dict[worker_info.id]
        cache = self.local_cache if worker_info is None else self.worker_cache_dict[worker_info.id]

        if self.in_current_cache(idx, cache):
            return self.get_from_cache(idx, cache)
        else:
            if hfd5_file is None:
                with h5py.File(self.file_path, "r") as hfd5_file:
                    self.update_cache(hfd5_file, idx, cache)
            else:
                self.update_cache(hfd5_file, idx, cache)
            return self.get_from_cache(idx, cache)

    def get_from_cache(self, idx, cache):
        cache_idx = idx - cache["current_cache_from"]
        try:
            if self.with_file_paths:
                return (cache["features_a"][cache_idx], cache["features_b"][cache_idx]), cache["labels"][cache_idx], (
                    cache["paths_a"][cache_idx], cache["paths_b"][cache_idx])
            return (cache["features_a"][cache_idx], cache["features_b"][cache_idx]), cache["labels"][cache_idx]
        except IndexError:
            raise IndexError(f'index out of range in cache, idx {idx} not in {cache["current_cache_from"]} - {cache["current_cache_to"]}, cache idx: {cache_idx}')

    @staticmethod
    def in_current_cache(idx, cache):
        if cache["current_cache_from"] is None:
            return False
        return cache["current_cache_from"] <= idx <= cache["current_cache_to"]

    def update_cache(self, hfd5_file, idx, cache):
        entry_idx = idx // self.batch_size
        entry = hfd5_file[f"entry_{entry_idx}"]
        cache["features_a"] = torch.tensor(entry["features_a"][()])
        cache["features_b"] = torch.tensor(entry["features_b"][()])
        cache["labels"] = torch.tensor(entry["labels"][()])
        cache["paths_a"] = entry.attrs["paths_a"]
        cache["paths_b"] = entry.attrs["paths_b"]
        cache["current_cache_from"] = entry_idx * self.batch_size
        cache["current_cache_to"] = cache["current_cache_from"] + entry["features_a"].shape[0] - 1


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.worker_file_dict[worker_id] = h5py.File(dataset.file_path, "r")
    dataset.worker_cache_dict[worker_id] = init_cache()
    # print(f"Worker {worker_id} initialized")


def create_vector_repr_dataloaders(root_dir='features_ext_vit', batch_size=32, bidirectional=False):
    if root_dir == '/home/daniel/data_dogs/vit_features_hdf5':
        train_dataset_augmented = FeaturesDatasetHDF5(f'{root_dir}/train_features_augmented.hdf5')
        validation_dataset = FeaturesDatasetHDF5(f'{root_dir}/validation_features.hdf5')
        test_dataset = FeaturesDatasetHDF5(f'{root_dir}/test_features.hdf5')
        train_feat_dataloader = DataLoader(train_dataset_augmented, batch_size=batch_size, shuffle=False, num_workers=6,
                                           worker_init_fn=worker_init_fn, prefetch_factor=4, pin_memory=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=6,
                                           worker_init_fn=worker_init_fn, prefetch_factor=4, pin_memory=True)
        test_feat_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6,
                                          worker_init_fn=worker_init_fn, prefetch_factor=4, pin_memory=True)
        return train_feat_dataloader, validation_dataloader, test_feat_dataloader
    else:
        train_dataset_augmented = FeaturesDataset(f'{root_dir}/train_features_augmented.pt',
                                                  bidirectional=bidirectional)
        validation_dataset = FeaturesDataset(f'{root_dir}/validation_features.pt')
        test_dataset = FeaturesDataset(f'{root_dir}/test_features.pt')
        train_feat_dataloader = DataLoader(train_dataset_augmented, batch_size=batch_size, shuffle=True, num_workers=12)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
        test_feat_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
        return train_feat_dataloader, validation_dataloader, test_feat_dataloader


def convert_to_hdf5(source_files, dest_path):
    import h5py
    import torch
    from tqdm import tqdm
    pbar = tqdm(total=None, unit=' batches', desc='Processing items (New Version) ', dynamic_ncols=True)
    with h5py.File(dest_path, "w") as f:
        entry_index = 0
        batch_size = None
        incomplete_batches = {
            "features_a": [],
            "features_b": [],
            "labels": [],
            "paths_a": [],
            "paths_b": []
        }
        for source_file in source_files:
            data = torch.load(source_file)
            for (feature_tensors_a, feature_tensors_b), labels, (path_str_a, path_str_b) in data:
                if batch_size is None:
                    batch_size = feature_tensors_a.shape[0]
                current_batch_size = feature_tensors_a.shape[0]
                if current_batch_size == batch_size:
                    grp = f.create_group(f"entry_{entry_index}")
                    grp.create_dataset("features_a", data=feature_tensors_a.numpy())
                    grp.create_dataset("features_b", data=feature_tensors_b.numpy())
                    grp.create_dataset("labels", data=labels)
                    grp.attrs["paths_a"] = path_str_a
                    grp.attrs["paths_b"] = path_str_b
                    entry_index += 1
                else:
                    incomplete_batches["features_a"].append(feature_tensors_a)
                    incomplete_batches["features_b"].append(feature_tensors_b)
                    incomplete_batches["labels"].append(labels)
                    incomplete_batches["paths_a"].extend(path_str_a)
                    incomplete_batches["paths_b"].extend(path_str_b)
                pbar.update(1)
            del data    

        incomplete_batches["features_a"] = torch.cat(incomplete_batches["features_a"])
        incomplete_batches["features_b"] = torch.cat(incomplete_batches["features_b"])
        incomplete_batches["labels"] = torch.cat(incomplete_batches["labels"])
        total_to_write = len(incomplete_batches["paths_a"])
        # write incomplete batches in batches of batch_size
        last_written = 0
        for i in range(0, total_to_write, batch_size):
            grp = f.create_group(f"entry_{entry_index}")
            grp.create_dataset("features_a", data=incomplete_batches["features_a"][i:i + batch_size].numpy())
            grp.create_dataset("features_b", data=incomplete_batches["features_b"][i:i + batch_size].numpy())
            grp.create_dataset("labels", data=incomplete_batches["labels"][i:i + batch_size])
            grp.attrs["paths_a"] = incomplete_batches["paths_a"][i:i + batch_size]
            grp.attrs["paths_b"] = incomplete_batches["paths_b"][i:i + batch_size]
            last_written = i + batch_size
            entry_index += 1

        # write last incomplete batch
        if last_written < total_to_write:
            grp = f.create_group(f"entry_{entry_index}")
            grp.create_dataset("features_a", data=incomplete_batches["features_a"][last_written:].numpy())
            grp.create_dataset("features_b", data=incomplete_batches["features_b"][last_written:].numpy())
            grp.create_dataset("labels", data=incomplete_batches["labels"][last_written:])
            grp.attrs["paths_a"] = incomplete_batches["paths_a"][last_written:]
            grp.attrs["paths_b"] = incomplete_batches["paths_b"][last_written:]

    pbar.close()
