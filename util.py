import os
import glob
import random
import cv2
import unittest
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from constants import *

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def draw_boxes(image, boxes):
    """
    Draw boxes on the given image and return the modified image
    """
    box_img = np.copy(image)
    for i, box in enumerate(boxes):
        x, y, w, h, c = box
        x_min = int(x - w / 2.0)
        y_min = int(y - h / 2.0)
        x_max = int(x + w / 2.0)
        y_max = int(y + h / 2.0)
        # Alternate the color of boxes so that they are easier to distinguish
        # box_color = None
        # if i % 3 == 0:
        #     box_color = (255, 0, 0)
        # elif i % 3 == 1:
        #     box_color = (0, 255, 0)
        # else:
        #     box_color = (0, 0, 255)
        cv2.rectangle(box_img, (x_min, y_min), (x_max, y_max), (0, 255 * c, 0), 3)
    return box_img

def bb_iou(box1, box2):
    """
    Calculate the IOU between two boxes
    """
    # Compute min and max of x and y values from both boxes
    x_min = min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
    x_max = max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
    y_min = min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
    y_max = max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)
    w1 = box1[2]
    h1 = box1[3]
    w2 = box2[2]
    h2 = box2[3]

    w_union = x_max - x_min
    h_union = y_max - y_min
    w_intersect = w1 + w2 - w_union
    h_intersect = h1 + h2 - h_union

    if w_intersect <= 0 or h_intersect <= 0:
        return 0.0

    # Area of intersection rectangle
    intersect_area = w_intersect * h_intersect

    # Compute area of prediction and ground truth boxes
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Compute intersection over union
    iou = intersect_area / float(box1_area + box2_area - intersect_area)
    return iou

def convert_lists_to_matrix(list_label_set):
    """
    Converts a list of sets/lists of labels to matrix form
    [x, y, w, h, c] -> matrix form
    """
    label_matrix = np.zeros((len(list_label_set), S1, S2, T))

    for i, label_set in enumerate(list_label_set):
        img = label_matrix[i]
        # Box and label are synonymous at this point
        for j, box in enumerate(label_set):
            x, y, w, h, c = box
            row = y // GRID_HEIGHT
            col = x // GRID_WIDTH
            
            # Check out of bounds
            if row < 0 or row >= S1 or col < 0 or col >= S2:
                continue

            # Grid cell top left position
            cell_x = col * GRID_WIDTH
            cell_y = row * GRID_HEIGHT
            # B * (5 + C) vector
            box_data = img[row, col]
            # Relative x from the top left corner of the grid cell
            box_data[0] = x - cell_x
            # Relative y
            box_data[1] = y - cell_y
            # Width
            box_data[2] = w
            # Height
            box_data[3] = h
            # Confidence level
            box_data[4] = 1

    # Relative x and y are normalized by grid cell size
    label_matrix[:, :, :, 0::5] = label_matrix[:, :, :, 0::5] / GRID_WIDTH
    label_matrix[:, :, :, 1::5] = label_matrix[:, :, :, 1::5] / GRID_HEIGHT
    # Normalize bounding box width/height by image width/height
    label_matrix[:, :, :, 2::5] = label_matrix[:, :, :, 2::5] / IMG_WIDTH
    label_matrix[:, :, :, 3::5] = label_matrix[:, :, :, 3::5] / IMG_HEIGHT

    return label_matrix

def convert_dict_to_matrix(label_dict):
    """
    Takes in bounding boxes (dictionary) extracted from processed data and converts them into a label matrix.
    """
    labels = np.zeros((len(label_dict), S1, S2, T))
    for i, (img_id, boxes) in enumerate(label_dict.items()):
        img = labels[i]
        for j, box in enumerate(boxes):
            x, y, w, h, c = box

            row = y // GRID_HEIGHT
            col = x // GRID_WIDTH
            
            # Check out of bounds
            if row < 0 or row >= S1 or col < 0 or col >= S2:
                continue

            # Grid cell top left position
            cell_x = col * GRID_WIDTH
            cell_y = row * GRID_HEIGHT
            # B * (5 + C) vector
            box_data = img[row, col]
            # Relative x from the top left corner of the grid cell
            box_data[0] = x - cell_x
            # Relative y
            box_data[1] = y - cell_y
            # Width
            box_data[2] = w
            # Height
            box_data[3] = h
            # Confidence level
            box_data[4] = 1

    # Relative x and y are normalized by grid cell size
    labels[:, :, :, 0::5] = labels[:, :, :, 0::5] / GRID_WIDTH
    labels[:, :, :, 1::5] = labels[:, :, :, 1::5] / GRID_HEIGHT
    # Normalize bounding box width/height by image width/height
    labels[:, :, :, 2::5] = labels[:, :, :, 2::5] / IMG_WIDTH
    labels[:, :, :, 3::5] = labels[:, :, :, 3::5] / IMG_HEIGHT

    return labels

def convert_matrix_to_dict(labels, conf_thresh=CONFIDENCE_THRESHOLD):
    """
    Takes in a label matrix and converts it into a label dictionary
    """
    label_dict = {}

    for i in range(len(labels)):
        label_dict[i] = []

    for l, label in enumerate(labels):
        num_grid_rows = label.shape[0]
        num_grid_cols = label.shape[1]

        # max_c = np.max(sigmoid(label[:, :, 4::5]))
        # min_c = np.min(sigmoid(label[:, :, 4::5]))
        max_c = np.max(label[:, :, 4::5])
        min_c = np.min(label[:, :, 4::5])
        print('MAX C: {}'.format(max_c))
        print('MIN C: {}'.format(min_c))

        for row in range(num_grid_rows):
            for col in range(num_grid_cols):
                box = label[row, col]
                for k in range(0, T, 5):
                    # c = sigmoid(box[k + 4])
                    c = box[k + 4]
                    # Skip grid if conf prob is less than the threshold
                    if c < conf_thresh:
                        continue
                    # x = sigmoid(box[k])
                    # y = sigmoid(box[k + 1])
                    x = box[k]
                    y = box[k + 1]
                    # w = sigmoid(box[k + 2])
                    # h = sigmoid(box[k + 3])
                    w = box[k + 2]
                    h = box[k + 3]
                    cell_topleft_x = col * GRID_WIDTH
                    cell_topleft_y = row * GRID_HEIGHT
                    # Unnormalize values
                    x = int(cell_topleft_x + (x * GRID_WIDTH))
                    y = int(cell_topleft_y + (y * GRID_HEIGHT))
                    w = int(w * IMG_WIDTH)
                    h = int(h * IMG_HEIGHT)
                    label_dict[l].append([x, y, w, h, c])

    return label_dict

def build_or_load(model_dir=MODEL_DIR, allow_load=True):
    """
    Loads model weights if model exists
    """
    from model import build_model
    model = build_model()
    if allow_load:
        try:
            model.load_weights(model_dir)
            print('Loaded model from file.')
        except:
            print('Unable to load model from file.')
    return model

def crop_images(old_dir, new_dir):
    """
    Crop images in specified directory to the standard size of 1000x700 and save them into the new directory
    """
    for path in tqdm(glob.glob(os.path.join(old_dir, '*.png'))):
        img = Image.open(path)
        filename = path.split('\\')[1]
        cropped_img = img.crop((0, 0, 1000, 700))
        cropped_img.save(new_dir + '/' + filename, 'PNG')


class TestUtilFunctions(unittest.TestCase):
    def test_map_matrix_conversion(self):
        label_dict = {
            '0020': [
                [676, 338, 65, 237, 0],
                [404, 285, 65, 237, 1]
            ],
            '0140': [
                [145, 242, 60, 253, 1]
            ]
        }

        expected = {
            0: [
                [404, 285, 65, 237, 1.0],
                [676, 338, 65, 237, 1.0]
            ],
            1: [
                [145, 242, 60, 253, 1.0]
            ]
        }

        self.assertEqual(convert_matrix_to_dict(convert_dict_to_matrix(label_dict)), expected)


if __name__ == '__main__':
    """
    tfile = open('data/image_boxes.txt', 'r')
    content = tfile.read()
    boxes = eval(content)

    # IOU Test
    tbox1 = [0, 0, 4, 4]
    tbox2 = [5, 5, 4, 4]
    pbox1 = [7, 7, 4, 4]
    pbox2 = [5, 5, 5, 5]
    true_boxes = np.array([[[tbox1, tbox2]]])
    pred_boxes = np.array([[[pbox1, pbox2]]])
    print('true boxes shape: ', true_boxes.shape)
    print('pred boxes shape: ', pred_boxes.shape)
    iou = calculate_iou(true_boxes, pred_boxes)
    print('IOU: ', iou)
    
    iou1 = bb_iou(tbox1, pbox1)
    iou2 = bb_iou(tbox2, pbox2)
    print('Individual IOU 1: ', iou1)
    print('Individual IOU 2: ', iou2)
    """

    # Run unit tests
    unittest.main()
