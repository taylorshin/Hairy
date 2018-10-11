import math
import random
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from constants import *

# tfile = open('image_boxes.txt', 'r')
# content = tfile.read()
# real_content = eval(content)
# print(real_content)

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
            self.data = np.asarray(hfile[keys[0]], 'float32')
        except Exception as e:
            print('Unable to load the data.', e)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def get_loader(filename):
    ds = HairFollicleDataset(filename)
    return torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        drop_last=True
    )

def validation_split(ds, split=0.2):
    """
    Generic function that takes a DataLoader object and splits it into training and validation datasets
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

loader = get_loader('data.hdf5')
train_set, val_set = validation_split(loader)
print('train set: ', len(train_set))
print('val set: ', len(val_set))
