import argparse
import random
import os
import time
import math
import h5py
import numpy as np
import matplotlib.pyplot as plt
from constants import *
from util import *

def load_data():
    """
    Load all of the images as matrices
    """
    data = []
    targets = []

    try:
        hfile = h5py.File('data/data.hdf5', 'r')
        keys = sorted(list(hfile.keys()), key=lambda k: int(k))
        # TODO: add back empty arrays for img 33 and 173 in txt file
        # CROP 2 PIXELS FROM LEFT AND RIGHT
        data = np.asarray([(hfile[k])[:, 2:-2, :] for k in keys], 'float32')
        # data = np.transpose(data, (0, 3, 1, 2))
        # Normalize pixels values to [0, 1]
        data = data / 255
        # Remove redundant channels because images are grayscale
        data = data[:, :, :, [0]]

        # Prepare target data
        tfile = open('data/image_boxes.txt', 'r')
        content = tfile.read()
        box_dict = eval(content)
        targets = convert_map_to_matrix(box_dict)
    except Exception as e:
            print('Unable to load the data.', e)

    return data, targets

def validation_split(data_xy, split=0.2):
    """
    Splits original dataset and into training and validation datasets
    """
    data, targets = data_xy

    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(14)
    r = list(range(len(targets)))
    # random.shuffle(r)

    val_size = int(math.ceil(len(r) * split))
    train_indicies = r[:-val_size]
    val_indicies = r[-val_size:]

    assert len(val_indicies) == val_size
    assert len(train_indicies) == len(r) - val_size

    train_data = np.array([data[i] for i in train_indicies])
    val_data = np.array([data[i] for i in val_indicies])
    train_targets = np.array([targets[i] for i in train_indicies])
    val_targets = np.array([targets[i] for i in val_indicies])

    return (train_data, train_targets), (val_data, val_targets)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=DATA_DIR, help='Directory with the hair follicle dataset')
    parser.add_argument('--out_dir', default=OUT_DIR, help='Where to write the new data')
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), 'Could not find the dataset at {}'.format(args.data_dir)

    train_set, val_set = validation_split(load_data())
    data, targets = train_set
    print('DATA: ', data)
    print('TARGETS: ', targets)
    data = np.squeeze(data)
    # print(val_set)
    # img = np.transpose(data[0], (1, 2, 0))
    img = data[0]
    # img = img[:, :, 0]
    print('img: ', img.shape, img)
    plt.imshow(img)
    plt.show()
