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

def shift_img(img, dx, dy):
    """
    Shift image by dx pixels horizontally and dy pixels vertically
    and fill the void parts of the image with 0
    """
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

    return img

def shift_labels(labels, dx, dy):
    """
    Shift a set of labels by dx pixels horizontally and dy pixels vertically
    """
    # Code commented out doesn't work for some reason...
    # shifted_labels = np.array(labels)
    # shifted_labels[:, 0] += dx
    # shifted_labels[:, 1] += dy
    # return shifted_labels.tolist()
    shifted_labels = []
    for label in labels:
        x, y, w, h, c = label
        label = [x + dx, y + dy, w, h, c]
        shifted_labels.append(label)
    return shifted_labels

def load_train_set(data_dirs, label_dirs, enable_data_aug=False):
    """
    Load the images and labels of the train set
    """
    data = []
    labels = []

    for data_dir, label_dir in zip(data_dirs, label_dirs):
        aug_indices = []
        shift_amounts = []

        if enable_data_aug:
            # Randomly select img/label indices to augment
            aug_indices = []
            samples = np.random.uniform(size=NUM_ITEMS_PER_DIR)
            for i, s in enumerate(samples):
                if s < DATA_AUG_PROB:
                    aug_indices.append(i)

            # Set shift amounts
            for i in range(len(aug_indices)):
                dx = random.randint(-PIXEL_SHIFT_X, PIXEL_SHIFT_X)
                dy = random.randint(0, PIXEL_SHIFT_Y)
                shift_amounts.append((dx, dy))

        ### Load data ###
        data.append(load_3d_data(data_dir, aug_indices, shift_amounts))

        ### Load labels ###
        labels.append(load_labels(label_dir, aug_indices, shift_amounts))

    data = np.concatenate(data)
    labels = np.concatenate(labels)

    # The mean and std of normalized pixel values of all training images
    # print('MEAN: {}, STD: {}, MIN: {}, MAX: {}'.format(np.mean(data), np.std(data), np.min(data), np.max(data)))

    return data, labels

# TODO: validation set loader
# def load_val_set():

# TODO: test set loader
# def load_test_set():

def load_image(filename):
    """
    Load an image from a file
    """
    img = Image.open(filename)
    img.load()
    img_data = np.array(img, dtype='float32')
    return img_data

def load_3d_data(data_dir, aug_indices=[], shift_amounts=[]):
    """
    Load all the 3d data by concatenating 2d slices
    """
    # Allow data augmentation only in train mode
    is_train = len(aug_indices) > 0

    filenames = os.listdir(data_dir)
    filenames = [os.path.join(data_dir, f) for f in filenames if f.endswith('.png')]
    # Parallel process all of the image loading and concatenating
    imgs = Parallel(n_jobs=multiprocessing.cpu_count(), prefer='threads')(delayed(load_image)(f) for f in filenames)
    imgs = np.array(imgs)

    data = []
    # TODO: Remove this exclusion of the first image/label which doesn't have context frames before
    for a, i in enumerate(range(LABEL_FRAME_INTERVAL, imgs.shape[0], LABEL_FRAME_INTERVAL)):
        if is_train and a in aug_indices:
            # Augment the images in the aug_indices list
            index = aug_indices.index(a)
            dx, dy = shift_amounts[index]
            aug_img = shift_img(imgs[i, :, :], dx, dy)
            # print('dx: {}, dy: {}'.format(dx, dy))
            # plt.imshow(img)
            # plt.show()
            aug_img = np.expand_dims(aug_img, axis=0)

            # TODO: Make function that augments list of images
            ### Also augment the rest of the frames in the volume PART 1 ###
            frames_before = imgs[i-CONTEXT_FRAMES:i, :, :]
            aug_frames_before = []
            for frame in frames_before:
                aug_frame = shift_img(frame, dx, dy)
                aug_frame = np.expand_dims(aug_frame, axis=0)
                aug_frames_before.append(aug_frame)
            aug_frames_before = np.vstack(aug_frames_before)

            ### Also augment the rest of the frames in the volume PART 2 ###
            frames_after = imgs[i+1:i+CONTEXT_FRAMES+1, :, :]
            aug_frames_after = []
            for frame in frames_after:
                aug_frame = shift_img(frame, dx, dy)
                aug_frame = np.expand_dims(aug_frame, axis=0)
                aug_frames_after.append(aug_frame)
            aug_frames_after = np.vstack(aug_frames_after)

            volume = np.concatenate((aug_frames_before, aug_img, aug_frames_after))
            volume = np.transpose(volume, (1, 2, 0))
            data.append(volume)
        else:
            # No data augmentation
            volume = imgs[i-CONTEXT_FRAMES:i+CONTEXT_FRAMES+1, :, :]
            volume = np.transpose(volume, (1, 2, 0))
            data.append(volume)

    # Normalize pixels values to [0, 1]
    return np.array(data) / 255

def load_labels(label_dir, aug_indices=[], shift_amounts=[]):
    """
    Load the labels for 3D data
    """
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

    # Convert label dictionary to matrix to feed into network
    labels = convert_map_to_matrix(aug_box_dict, False)
    return labels

def load_2d_data():
    """
    Load all of the images as matrices
    """
    data = []
    labels = []

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
        labels = convert_map_to_matrix(box_dict)
    except Exception as e:
            print('Unable to load the data.', e)

    return data, labels

def validation_split(data_xy, split=0.2):
    """
    Splits original dataset and into training and validation datasets
    """
    data, labels = data_xy

    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(14)
    r = list(range(len(labels)))
    # random.shuffle(r)

    val_size = int(math.ceil(len(r) * split))
    train_indicies = r[:-val_size]
    val_indicies = r[-val_size:]

    assert len(val_indicies) == val_size
    assert len(train_indicies) == len(r) - val_size

    train_data = np.array([data[i] for i in train_indicies])
    val_data = np.array([data[i] for i in val_indicies])
    train_labels = np.array([labels[i] for i in train_indicies])
    val_labels = np.array([labels[i] for i in val_indicies])

    return (train_data, train_labels), (val_data, val_labels)

class TestUtilFunctions(unittest.TestCase):

    def test_shift_img(self):
        img = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ]

        expected = [
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ]

        result = shift_img(img, 2, 1).tolist()

        self.assertCountEqual(result, expected)

    def test_shift_labels(self):
        labels = [
            [460, 196, 72, 250, 2],
            [248, 196, 73, 251, 2],
            [896, 172, 74, 251, 2]
        ]

        expected = [
            [510, 296, 72, 250, 2],
            [298, 296, 73, 251, 2],
            [946, 272, 74, 251, 2]
        ]

        result = shift_labels(labels, 50, 100)

        self.assertCountEqual(result, expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=DATA_DIR, help='Directory with the hair follicle dataset')
    parser.add_argument('--out_dir', default=OUT_DIR, help='Where to write the new data')
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), 'Could not find the dataset at {}'.format(args.data_dir)

    # Test the shifting of images/labels
    # dx = random.randint(-PIXEL_SHIFT_X, PIXEL_SHIFT_X)
    # dy = random.randint(0, PIXEL_SHIFT_Y)
    # print('dx: {}, dy: {}'.format(dx, dy))
    # index = 1
    # data = load_3d_data('data/J_data', [index], [(dx, dy)])
    # labels = load_labels('data/labels/image_boxes_J.txt', [index], [(dx, dy)])
    # img = data[index, :, :, 5]
    # img = img[:, :, np.newaxis]
    # img = np.tile(img, (1, 1, 3))
    # boxes = convert_matrix_to_map(labels, conf_thresh=CONFIDENCE_THRESHOLD)
    # box_img = draw_boxes(img, boxes[index])
    # box_img = np.squeeze(box_img)
    # plt.imshow(box_img)
    # plt.show()

    # data, labels = load_train_set(['data/H_data', 'data/I_data'], ['data/labels/image_boxes_H.txt', 'data/labels/image_boxes_I.txt'])
    # print('data: ', data.shape, data)
    # print('labels: ', labels.shape, labels)

    # Run unit tests
    unittest.main()
