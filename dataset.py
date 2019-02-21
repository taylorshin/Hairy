import argparse
import random
import os
import time
import math
import h5py
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm
from constants import *
from util import *

def load_train_set(data_dirs, label_dirs):
    data = []
    labels = []
    for d in data_dirs:
        data.append(load_3d_data(d))
    for d in label_dirs:
        labels.append(load_labels(d))
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    return data, labels

# def load_val_set():

# def load_test_set():

def load_image(filename):
    img = Image.open(filename)
    img.load()
    img_data = np.array(img, dtype='float32')
    return img_data

def load_3d_data(data_dir):
    """
    Load all the 3d data by concatenating 2d slices
    """
    filenames = os.listdir(data_dir)
    filenames = [os.path.join(data_dir, f) for f in filenames if f.endswith('.png')]
    # Parallel process all of the image loading and concatenating
    imgs = Parallel(n_jobs=multiprocessing.cpu_count(), prefer='threads')(delayed(load_image)(f) for f in filenames)
    imgs = np.array(imgs)
    # imgs = np.transpose(imgs, (1, 2, 0))
    data = []
    for i in range(20, imgs.shape[0], 20):
        volume = imgs[i-5:i+5+1, :, :]
        volume = np.transpose(volume, (1, 2, 0))
        data.append(volume)
    # Normalize pixels values to [0, 1]
    return np.array(data) / 255

def load_labels(dir):
    tfile = open(dir, 'r')
    content = tfile.read()
    box_dict = eval(content)
    targets = convert_map_to_matrix(box_dict, False)
    return targets[1:]

def load_2d_data():
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

    """
    train_set, val_set = validation_split(load_2d_data())
    data, targets = train_set
    data = np.squeeze(data)
    # img = np.transpose(data[0], (1, 2, 0))
    img = data[0]
    # img = img[:, :, 0]
    plt.imshow(img)
    plt.show()
    """

    # data = load_3d_data('data/H_data')
    # print('Data: ', data.shape)
    # targets = load_labels('data/labels/image_boxes_G.txt')
    # print('Targets: ', targets.shape)

    data, labels = load_train_set(['data/H_data', 'data/I_data'], ['data/labels/image_boxes_H.txt', 'data/labels/image_boxes_I.txt'])
    print('data: ', data.shape, data)
    # print('labels: ', labels.shape, labels)
    
