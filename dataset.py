import time
import math
import random
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from constants import *

class GenHelper(Dataset):
    def __init__(self, parent, length, mapping):
        # Mapping from this index to the parent dataset index
        self.mapping = mapping
        self.length = length
        self.parent = parent

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.parent[self.mapping[index]]

class HairFollicleDataset(Dataset):
    def __init__(self, filename):
        self.data = []
        try:
            hfile = h5py.File(filename, 'r')
            keys = list(hfile.keys())
            # start = time.time()
            # Value returns ndarray from HDF5 data. It also speeds up data loading.
            self.data = np.asarray([hfile[k].value for k in keys], 'float32')
            # end = time.time()
            # print('Time elapsed: ', end - start)
            # plt.imshow(self.data[0])
            # plt.show()
            self.data = np.transpose(self.data, (0, 3, 1, 2))
        except Exception as e:
            print('Unable to load the data.', e)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def get_tv_loaders(filename, batch_size):
    ds = HairFollicleDataset(filename)
    train_set, val_set = validation_split(ds)
    return get_loader(train_set, batch_size), get_loader(val_set, batch_size)

def get_loader(ds, batch_size):
    return torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True
    )

def validation_split(ds, split=0.2):
    """
    Generic function that takes a DataLoader object and splits it into training and validation dataloaders
    """
    r = list(range(len(ds)))
    random.shuffle(r)

    val_size = int(math.ceil(len(r) * split))
    train_indicies = r[:-val_size]
    val_indicies = r[-val_size:]

    assert len(val_indicies) == val_size
    assert len(train_indicies) == len(r) - val_size

    train_set = GenHelper(ds, len(r) - val_size, train_indicies)
    val_set = GenHelper(ds, val_size, val_indicies)

    return train_set, val_set

if __name__ == '__main__':
    train_loader, val_loader = get_tv_loaders('data.hdf5', 16)
    print('val loader: ', val_loader)
    for data in val_loader:
        # NOTE: printing data from a dataloader is slow
        print(data)
