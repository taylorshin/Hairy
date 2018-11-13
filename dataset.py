import time
import math
import random
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from constants import *
from util import *

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
        self.labels = []

        try:
            hfile = h5py.File(filename, 'r')
            keys = list(hfile.keys())
            # TODO: add back empty arrays for img 33 and 173 in txt file
            # start = time.time()
            # Value returns ndarray from HDF5 data. It also speeds up data loading.
            # CROP 2 PIXELS FROM LEFT AND RIGHT
            self.data = np.asarray([hfile[k].value[:, 2:-2, :] for k in keys], 'float32')
            self.data = np.transpose(self.data, (0, 3, 1, 2))
            # end = time.time()
            # print('Time elapsed: ', end - start)
            # plt.imshow(self.data[0])
            # plt.show()
            tfile = open('image_boxes.txt', 'r')
            content = tfile.read()
            box_dict = eval(content)
            self.labels = convert_map_to_matrix(box_dict, S1, S2, T_temp)
        except Exception as e:
            print('Unable to load the data.', e)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

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

def convert_map_to_matrix(box_dict, width, height, depth):
    """
    Takes in bounding boxes (dict) extracted from processed data and converts them into a label matrix.
    """
    n = len(box_dict)
    labels = np.zeros((n, width, height, depth))
    # TODO: Remove this special code for missing data in old dataset
    i = 0
    # for i, boxes in box_dict.items():
    for _, boxes in box_dict.items():
        # img = labels[i - 1]
        img = labels[i]
        for j, box in enumerate(boxes):
            x, y, w, h = box
            row = y // GRID_HEIGHT
            col = x // GRID_WIDTH
            # Grid cell center position
            cell_x = col * GRID_WIDTH + GRID_WIDTH / 2
            cell_y = row * GRID_HEIGHT + GRID_HEIGHT / 2
            # B * (5 + C) vector
            box_data = img[row, col]
            # Relative x from the center of the grid cell
            box_data[j * 5] = cell_x - x
            # Relative y
            box_data[j * 5 + 1] = cell_y - y
            # Width
            box_data[j * 5 + 2] = w
            # Height
            box_data[j * 5 + 3] = h
            # Confidence level
            box_data[j * 5 + 4] = 1
        # TODO: Remove this special code for missing data in old dataset
        i += 1
    # Relative x's are normalized by half the grid cell size
    labels[:, :, :, 0::5] = labels[:, :, :, 0::5] / (GRID_WIDTH / 2)
    labels[:, :, :, 1::5] = labels[:, :, :, 1::5] / (GRID_HEIGHT / 2)
    # Width and height are normalized by the corresponding max values (set to 500 for now)
    labels[:, :, :, 2::5] = labels[:, :, :, 2::5] / MAX_BOX_WIDTH
    labels[:, :, :, 3::5] = labels[:, :, :, 3::5] / MAX_BOX_HEIGHT

    return labels

def convert_matrix_to_map(labels):
    box_dict = {}
    for i in range(len(labels)):
        box_dict[i] = []

    for l, label in enumerate(labels):
        num_grid_rows = label.shape[0]
        num_grid_cols = label.shape[1]

        for row in range(num_grid_rows):
            for col in range(num_grid_cols):
                box_vals = label[row, col]
                for k in range(0, T, 5):
                    c = box_vals[k + 4]
                    if c <= CONFIDENCE_THRESHOLD:
                        continue
                    x = box_vals[k]
                    y = box_vals[k + 1]
                    w = box_vals[k + 2]
                    h = box_vals[k + 3]
                    cell_topleft_x = col * GRID_WIDTH
                    cell_topleft_y = row * GRID_HEIGHT
                    cell_center_x = cell_topleft_x + GRID_WIDTH / 2
                    cell_center_y = cell_topleft_y + GRID_HEIGHT / 2
                    # Unnormalize values
                    # x = int(cell_center_x - (x * (GRID_WIDTH / 2)))
                    # y = int(cell_center_y - (y * (GRID_HEIGHT / 2)))
                    # w = int(w * MAX_BOX_WIDTH)
                    # h = int(h * MAX_BOX_HEIGHT)
                    box_dict[l].append([x, y, w, h])
    return box_dict

def transform_network_output(output):
    print('before: ', output)
    for i, img in enumerate(output):
        num_grid_rows = img.shape[0]
        num_grid_cols = img.shape[1]

        for row in range(num_grid_rows):
            for col in range(num_grid_cols):
                box_vals = img[row, col]
                for k in range(0, T, 5):
                    x = box_vals[k]
                    y = box_vals[k + 1]
                    w = box_vals[k + 2]
                    h = box_vals[k + 3]
                    c = box_vals[k + 4]

                    cell_topleft_x = col * GRID_WIDTH
                    cell_topleft_y = row * GRID_HEIGHT
                    cell_center_x = cell_topleft_x + GRID_WIDTH / 2
                    cell_center_y = cell_topleft_y + GRID_HEIGHT / 2

                    # Transform network outputs
                    if x > 0:
                        x = sigmoid(x) + cell_topleft_x
                        box_vals[k] = x
                    if y > 0:
                        y = sigmoid(y) + cell_topleft_y
                        box_vals[k + 1] = y
                    if w > 0:
                        w = MAX_BOX_WIDTH * math.exp(w)
                        box_vals[k + 2] = w
                    if h > 0:
                        h = MAX_BOX_HEIGHT * math.exp(h)
                        box_vals[k + 3] = h
                    if c > 0:
                        c = sigmoid(c)
                        box_vals[k + 4] = c
    print('after: ', output)
    return output

if __name__ == '__main__':
    # train_loader, val_loader = get_tv_loaders('data.hdf5', 16)
    # print('val loader: ', val_loader)
    # for data in val_loader:
    #     # NOTE: printing data from a dataloader is slow
    #     print(data)
    tfile = open('image_boxes.txt', 'r')
    content = tfile.read()
    box_dict = eval(content)
    labels = convert_map_to_matrix(box_dict, S1, S2, T_temp)
    print('labels: ', labels.shape)
    # print(labels[0, 3, 10])
    # print(labels[0, 3, 8])
    print('first label: ', labels[0, 3])
    boxes = convert_matrix_to_map(labels)
    print(boxes)
    
    ds = HairFollicleDataset('data.hdf5')
    index = 0
    image = np.transpose(ds[index][0], (1, 2, 0))
    boxed_image = draw_boxes(image, boxes[index])
    plt.imshow(boxed_image)
    plt.show()
