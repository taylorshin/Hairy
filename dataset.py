import argparse
import random
import os
import time
import math
import h5py
import unittest
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm
from constants import *
from util import *

def shift_image_and_label(img, label, dx, dy):
    """
    Data augmentation: shift image and its label
    """
    # Shift image
    img = np.roll(img, dy, axis=0)
    img = np.roll(img, dx, axis=1)

    if dy > 0:
        img[:dy, :] = 0
    elif dy < 0:
        img[dy:, :] = 0
    if dx > 0:
        img[:, :dx] = 0
    elif dx < 0:
        img[:, dx:] = 0

    # plt.imshow(img)
    # plt.show()

    # Shift label
    print('label: ', label)
    x, y, w, h, c = label
    label = [x + dx, y + dy, w, h, c]

    return img, label

def shift_img(img, dx, dy):
    img = np.roll(img, dy, axis=0)
    img = np.roll(img, dx, axis=1)

    if dy > 0:
        img[:dy, :] = 0
    elif dy < 0:
        img[dy:, :] = 0
    if dx > 0:
        img[:, :dx] = 0
    elif dx < 0:
        img[:, dx:] = 0

    # plt.imshow(img)
    # plt.show()

    return img

def shift_labels(labels, dx, dy):
    shifted_labels = []
    for label in labels:
        x, y, w, h, c = label
        label = [x + dx, y + dy, w, h, c]
        shifted_labels.append(label)

    return shifted_labels

def load_train_set(data_dirs, label_dirs):
    data = []
    labels = []

    for data_dir, label_dir in zip(data_dirs, label_dirs):
        # Randomly select img/label indices to augment
        aug_indices = []
        samples = np.random.uniform(size=NUM_ITEMS_PER_DIR)
        for i, s in enumerate(samples):
            if s < AUG_PROB:
                aug_indices.append(i)

        # print('aug indices: ', aug_indices)

        # Set shift amounts
        shift_amounts = []
        for i in range(len(aug_indices)):
            dx = random.randint(-PIXEL_SHIFT_X, PIXEL_SHIFT_X)
            dy = random.randint(0, PIXEL_SHIFT_Y)
            shift_amounts.append((dx, dy))

        # print('shift amounts: ', shift_amounts)

        ### Load data ###
        data.append(load_3d_data(data_dir, aug_indices, shift_amounts))

        ### Load labels ###
        labels.append(load_labels(label_dir, aug_indices, shift_amounts))

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

def load_3d_data(data_dir, aug_indices=[], shift_amounts=[]):
    """
    Load all the 3d data by concatenating 2d slices
    """
    # Train mode
    is_train = len(aug_indices) > 0

    filenames = os.listdir(data_dir)
    filenames = [os.path.join(data_dir, f) for f in filenames if f.endswith('.png')]
    # Parallel process all of the image loading and concatenating
    imgs = Parallel(n_jobs=multiprocessing.cpu_count(), prefer='threads')(delayed(load_image)(f) for f in filenames)
    imgs = np.array(imgs)
    # imgs = np.transpose(imgs, (1, 2, 0))
    data = []
    # TODO: Remove this exclusion of the first image/label which doesn't have context frames before
    j = 0
    for i in range(LABEL_FRAME_INTERVAL, imgs.shape[0], LABEL_FRAME_INTERVAL):
        # Randomly augment image
        if is_train and j in aug_indices:
            index = aug_indices.index(j)
            dx, dy = shift_amounts[index]
            img = shift_img(imgs[i, :, :], dx, dy)
            # print('dx: {}, dy: {}'.format(dx, dy))
            # plt.imshow(img)
            # plt.show()
            img = np.expand_dims(img, axis=0)
            frames_before = imgs[i-CONTEXT_FRAMES:i, :, :]
            frames_after = imgs[i+1:i+CONTEXT_FRAMES+1, :, :]
            volume = np.concatenate((frames_before, img, frames_after))
            volume = np.transpose(volume, (1, 2, 0))
            data.append(volume)
        else:
            volume = imgs[i-CONTEXT_FRAMES:i+CONTEXT_FRAMES+1, :, :]
            volume = np.transpose(volume, (1, 2, 0))
            data.append(volume)
        j += 1
    # Normalize pixels values to [0, 1]
    return np.array(data) / 255

def load_labels(label_dir, aug_indices=[], shift_amounts=[]):
    # Only do data augmentation during training
    is_train = len(aug_indices) > 0

    tfile = open(label_dir, 'r')
    content = tfile.read()
    box_dict = eval(content)
    # Sort dictionary by key because 3D data labels are out of order
    box_dict = dict(sorted(box_dict.items()))
    # TODO: Remove this exclusion of the first image/label which doesn't have context frames before
    del box_dict['0000']

    # Augment the labels whose indices are in the aug_indices list
    aug_box_dict = {}
    for i, key in enumerate(box_dict.keys()):
        if is_train and i in aug_indices:
            index = aug_indices.index(i)
            dx, dy = shift_amounts[index]
            # Shift all labels in image not just one
            labels = shift_labels(box_dict[key], dx, dy)
            aug_box_dict[key] = labels
        else:
            aug_box_dict[key] = box_dict[key]
    targets = convert_map_to_matrix(aug_box_dict, False)
    # TODO: Remove this exclusion of the first image/label which doesn't have context frames before
    # return targets[1:]
    return targets

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

class TestUtilFunctions(unittest.TestCase):
    def test_shift_image_and_label(self):
        img = np.reshape(np.arange(10), (2, 5))

        label = np.array([200, 300, 123, 456])

        expected_img = [
            [0, 0, 0, 1, 2],
            [0, 0, 5, 6, 7]
        ]

        expected_label = np.array([202, 300, 123, 456])

        out_img, out_label = shift_image_and_label(img, label, 2, 0)
        diff_img = out_img - expected_img
        diff_label = out_label - expected_label

        self.assertTrue(np.all(diff_img == 0))
        self.assertTrue(np.all(diff_label == 0))


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

    # Test the shifting of images/labels
    dx = random.randint(-PIXEL_SHIFT_X, PIXEL_SHIFT_X)
    dy = random.randint(0, PIXEL_SHIFT_Y)
    print('dx: {}, dy: {}'.format(dx, dy))
    index = 3
    data = load_3d_data('data/H_data', [index], [(dx, dy)])
    labels = load_labels('data/labels/image_boxes_H.txt', [index], [(dx, dy)])
    img = data[index, :, :, 5]
    img = img[:, :, np.newaxis]
    img = np.tile(img, (1, 1, 3))
    boxes = convert_matrix_to_map(labels, conf_thresh=CONFIDENCE_THRESHOLD)
    box_img = draw_boxes(img, boxes[index])
    box_img = np.squeeze(box_img)
    plt.imshow(box_img)
    plt.show()

    # data, labels = load_train_set(['data/H_data', 'data/I_data'], ['data/labels/image_boxes_H.txt', 'data/labels/image_boxes_I.txt'])
    # print('data: ', data.shape, data)
    # print('labels: ', labels.shape, labels)

    # Run unit tests
    # unittest.main()
