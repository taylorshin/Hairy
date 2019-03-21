import random
import os
import math
import h5py
import unittest
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import tensorflow as tf
from tensorflow import keras
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm
from constants import *
from util import *

class DataGenerator(keras.utils.Sequence):

    def __init__(self, data_dirs, label_files, batch_size, shuffle=True, enable_data_aug=True):
        print('Loading data...')
        # Load images
        data = []
        for data_dir in data_dirs:
            data.append(load_3d_data(data_dir))
        data = np.concatenate(data)

        # Load labels
        labels = get_label_list(label_files)
        
        # List of image/label indices
        indices = list(range(len(labels)))

        self.data = data
        self.labels = labels
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.enable_data_aug = enable_data_aug
        self.on_epoch_end()

    def __len__(self):
        ### Return the number of batches per epoch ###
        return int(np.ceil(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        ### Generate one batch of data ###
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        # Create placeholders for x and y so original data is not modified
        imgs = np.copy(self.data[indices])
        list_label_set = np.copy(self.labels[indices])

        # Randomly augment data/labels
        if self.enable_data_aug and np.random.random() < DATA_AUG_PROB:
            # dx = random.randint(-PIXEL_SHIFT_X, PIXEL_SHIFT_X)
            # dy = random.randint(0, PIXEL_SHIFT_Y)

            # # Augment batch of images
            # imgs = shift_img_batch(imgs, dx, dy)

            # # Augment batch of labels
            # aug_labels = []
            # for labels in list_label_set:
            #     aug_labels.append(shift_labels(labels, dx, dy))
            # list_label_set = aug_labels

            # Augment individual images in a batch
            for i in range(len(imgs)):
                dx = random.randint(-PIXEL_SHIFT_X, PIXEL_SHIFT_X)
                dy = random.randint(0, PIXEL_SHIFT_Y)
                # Augment volume of images
                imgs[i] = shift_volume(imgs[i], dx, dy)
                # Augment labels
                list_label_set[i] = shift_labels(list_label_set[i], dx, dy)

        x = np.array(imgs)
        x = np.transpose(x, (0, 3, 1, 2))
        x = np.expand_dims(x, axis=4)
        # print('X: ', x.shape)
        y = convert_lists_to_matrix(list_label_set)
        # y = np.transpose(y, (0, 3, 1, 2))
        # y = np.expand_dims(y, axis=4)
        # print('Y: ', y.shape)
        return x, y

    def on_epoch_end(self):
        ### Updates indices after each epoch ###
        self.indices = list(range(len(self.labels)))
        if self.shuffle:
            np.random.shuffle(self.indices)


def get_label_list(label_files):
    """
    Return a list of every label in each label file
    """
    labels_list = []
    for label_file in label_files:
        label_dict = load_label_dict(label_file)
        # Add each list of boxes to the overall labels list
        for key in label_dict.keys():
            labels = np.array(label_dict[key])

            # Downscale image
            if len(labels) > 0:
                labels[:, :4] = labels[:, :4] // DOWNSCALE_FACTOR

            labels_list.append(labels)
    return np.array(labels_list)

def shift_img_batch(batch, dx, dy):
    """
    Shift batch (8, 700, 1000, 11) of images by dx pixels horizontally and dy pixels vertically
    and fill the void parts of the image with the min pixel value
    """
    min_pixel = np.min(batch)
    batch = np.transpose(batch, (0, 3, 1, 2))
    batch = np.roll(batch, [dy, dx], axis=(2, 3))

    if dy > 0:
        batch[:, :, :dy, :] = min_pixel
    elif dy < 0:
        batch[:, :, dy:, :] = min_pixel
    if dx > 0:
        batch[:, :, :, :dx] = min_pixel
    elif dx < 0:
        batch[:, :, :, dx:] = min_pixel

    batch = np.transpose(batch, (0, 2, 3, 1))
    return batch

def shift_img(img, dx, dy):
    """
    Shift a single image by dx pixels horizontally and dy pixels vertically
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

def shift_volume(volume, dx, dy):
    """
    Shift volume of images
    """
    min_pixel = np.min(volume)
    volume = np.transpose(volume, (2, 0, 1))
    volume = np.roll(volume, [dy, dx], axis=(1, 2))

    if dy > 0:
        volume[:, :dy, :] = min_pixel
    elif dy < 0:
        volume[:, dy:, :] = min_pixel
    if dx > 0:
        volume[:, :, :dx] = min_pixel
    elif dx < 0:
        volume[:, :, dx:] = min_pixel

    volume = np.transpose(volume, (1, 2, 0))
    return volume

def shift_labels(labels, dx, dy):
    """
    Shift a set of labels (varied length) by dx pixels horizontally and dy pixels vertically
    """
    shifted_labels = []
    for label in labels:
        x, y, w, h, c = label
        label = [x + dx, y + dy, w, h, c]
        shifted_labels.append(label)
    return shifted_labels

def load_image(filename, size=(IMG_WIDTH, IMG_HEIGHT)):
    """
    Load an image from a file
    """
    img = Image.open(filename)
    # Resize image if specified
    img = img.resize(size, Image.ANTIALIAS)
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

    data = []
    # Exclude the first image/label which doesn't have context frames before
    for i in range(LABEL_FRAME_INTERVAL, imgs.shape[0], LABEL_FRAME_INTERVAL):
        volume = imgs[i-CONTEXT_FRAMES:i+CONTEXT_FRAMES+1, :, :]
        volume = np.transpose(volume, (1, 2, 0))
        data.append(volume)

    dataset_name = data_dir.split('/')[-1]
    print('{} - mean: {}, std: {}, min: {}, max: {}'.format(dataset_name, np.mean(data), np.std(data), np.min(data), np.max(data)))

    # Normalize pixels values to [0, 1]
    return (np.array(data) - np.min(data)) / np.max(data)

def load_label_dict(label_file):
    """
    Load dictionary of labels from a label file
    """
    tfile = open(label_file, 'r')
    content = tfile.read()
    label_dict = eval(content)
    # Sort dictionary by key because 3D data labels are out of order
    label_dict = dict(sorted(label_dict.items()))
    # Exclude the first image/label which doesn't have context frames before
    del label_dict['0000']
    return label_dict

def load_label_matrix(label_file):
    """
    Load the labels for 3D data
    """
    label_dict = load_label_dict(label_file)
    # Convert label dictionary to matrix to feed into network
    labels = convert_dict_to_matrix(label_dict)
    return labels

def verify_data_generator(generator):
    """
    Verifies that the data is correctly generated and augmented
    """
    for i, (data_batch, label_batch) in enumerate(generator):
        print('BATCH {}'.format(i))
        print('DATA BATCH: ', data_batch.shape)
        print('LABELS BATCH: ', label_batch.shape)
        for j in range(BATCH_SIZE):
            print('IMG {}'.format(j))
            img = data_batch[j, CONTEXT_FRAMES, ...]
            img = np.tile(img, (1, 1, 3))
            boxes = convert_matrix_to_dict(label_batch, conf_thresh=CONFIDENCE_THRESHOLD)
            box_img = draw_boxes(img, boxes[j])
            box_img = np.squeeze(box_img)
            plt.imshow(box_img)
            plt.show()

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

    def test_shift_img_batch(self):
        # (batch, frames, height, width)
        # (1, 4, 5, 2)
        batch = [
            [
                [
                    [1, 1],
                    [2, 2],
                    [3, 3],
                    [4, 4]
                ],
                [
                    [5, 5],
                    [6, 6],
                    [7, 7],
                    [8, 8]
                ],
                [
                    [9, 9],
                    [10, 10],
                    [11, 11],
                    [12, 12]
                ],
                [
                    [13, 13],
                    [14, 14],
                    [15, 15],
                    [16, 16]
                ]
            ]
        ]
        batch = np.array(batch)

        expected = [
            [
                [
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1]
                ],
                [
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1]
                ],
                [
                    [1, 1],
                    [1, 1],
                    [2, 2],
                    [3, 3]
                ],
                [
                    [1, 1],
                    [5, 5],
                    [6, 6],
                    [7, 7]
                ]
            ]
        ]

        result = shift_img_batch(batch, 1, 2)

        self.assertCountEqual(result.tolist(), expected)


if __name__ == '__main__':
    ### Run unit tests ###
    unittest.main()

    # Test DataGenerator class
    data_dirs = ['data/G_data']
    label_files = ['data/labels/image_boxes_G.txt']
    generator = DataGenerator(data_dirs, label_files)
    verify_data_generator(generator)

