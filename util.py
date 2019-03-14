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
    Draw boxes in the given image and return the modified image
    """
    for i, box in enumerate(boxes):
        x, y, w, h = box
        x_min = int(x - w / 2.0)
        y_min = int(y - h / 2.0)
        x_max = int(x + w / 2.0)
        y_max = int(y + h / 2.0)
        # Alternate the color of boxes so that they are easier to distinguish
        box_color = None
        if i % 3 == 0:
            box_color = (255, 0, 0)
        elif i % 3 == 1:
            box_color = (0, 255, 0)
        else:
            box_color = (0, 0, 255)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, 2)
    return image

def calculate_iou(true_boxes, pred_boxes):
    x_min = np.minimum(true_boxes[:, :, :, 0] - (true_boxes[:, :, :, 2] / 2.0), pred_boxes[:, :, :, 0] - (pred_boxes[:, :, :, 2] / 2.0))
    x_max = np.maximum(true_boxes[:, :, :, 0] + (true_boxes[:, :, :, 2] / 2.0), pred_boxes[:, :, :, 0] + (pred_boxes[:, :, :, 2] / 2.0))
    y_min = np.minimum(true_boxes[:, :, :, 1] - (true_boxes[:, :, :, 3] / 2.0), pred_boxes[:, :, :, 1] - (pred_boxes[:, :, :, 3] / 2.0))
    y_max = np.maximum(true_boxes[:, :, :, 1] + (true_boxes[:, :, :, 3] / 2.0), pred_boxes[:, :, :, 1] + (pred_boxes[:, :, :, 3] / 2.0))
    w1 = true_boxes[:, :, :, 2]
    h1 = true_boxes[:, :, :, 3]
    w2 = pred_boxes[:, :, :, 2]
    h2 = pred_boxes[:, :, :, 3]

    w_union = x_max - x_min
    h_union = y_max - y_min
    w_intersect = w1 + w2 - w_union
    h_intersect = h1 + h2 - h_union

    w_intersect = np.maximum(w_intersect, np.zeros(w_intersect.shape))
    h_intersect = np.maximum(h_intersect, np.zeros(h_intersect.shape))

    intersect_area = w_intersect * h_intersect

    true_boxes_area = w1 * h1
    pred_boxes_area = w2 * h2

    iou = intersect_area / (true_boxes_area + pred_boxes_area - intersect_area)

    return iou

def bb_iou(box1, box2):
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

def convert_map_to_matrix(box_dict, is_2d_data=True):
    """
    Takes in bounding boxes (dict) extracted from processed data and converts them into a label matrix.
    """
    labels = np.zeros((len(box_dict), S1, S2, T))
    for i, (img_id, boxes) in enumerate(box_dict.items()):
        img = labels[i]
        for j, box in enumerate(boxes):
            if is_2d_data:
                x, y, w, h = box
            else:
                x, y, w, h, c = box

            # Downscale image
            x //= DOWNSCALE_FACTOR
            y //= DOWNSCALE_FACTOR
            w //= DOWNSCALE_FACTOR
            h //= DOWNSCALE_FACTOR

            row = y // GRID_HEIGHT
            col = x // GRID_WIDTH
            
            # Check out of bounds
            if row < 0 or row >= S1 or col < 0 or col >= S2:
                continue

            # Grid cell center position
            # cell_x = col * GRID_WIDTH + GRID_WIDTH / 2
            # cell_y = row * GRID_HEIGHT + GRID_HEIGHT / 2
            # Grid cell top left position
            cell_x = col * GRID_WIDTH
            cell_y = row * GRID_HEIGHT
            # B * (5 + C) vector
            box_data = img[row, col]
            # Relative x from the center of the grid cell
            # box_data[j * 5] = cell_x - x
            # Relative x from the top left corner of the grid cell
            box_data[0] = x - cell_x
            # Relative y
            # box_data[j * 5 + 1] = cell_y - y
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

def convert_matrix_to_map(labels, conf_thresh=CONFIDENCE_THRESHOLD):
    box_dict = {}
    for i in range(len(labels)):
        box_dict[i] = []

    for l, label in enumerate(labels):
        num_grid_rows = label.shape[0]
        num_grid_cols = label.shape[1]

        # max_c = np.max(sigmoid(label[:, :, 4::5]))
        # min_c = np.min(sigmoid(label[:, :, 4::5]))
        max_c = np.max(label[:, :, 4::5])
        min_c = np.min(label[:, :, 4::5])
        print('MAX C: {}'.format(max_c))
        print('MIN C: {}'.format(min_c))

        # c_list = label[:, :, 4::5]
        # c_list = np.squeeze(c_list)
        # print('c list: ', c_list, c_list.shape)
        # top_c_list = c_list.argsort()[-5:][::-1]
        # print("Top C's: ", top_c_list)

        for row in range(num_grid_rows):
            for col in range(num_grid_cols):
                box_vals = label[row, col]
                for k in range(0, T, 5):
                    # c = sigmoid(box_vals[k + 4])
                    c = box_vals[k + 4]
                    # Skip grid if conf prob is less than the threshold
                    if c <= conf_thresh:
                        continue
                    # x = sigmoid(box_vals[k])
                    # y = sigmoid(box_vals[k + 1])
                    x = box_vals[k]
                    y = box_vals[k + 1]
                    # Width and height values are sometimes negative because of leaky relu
                    # w = sigmoid(box_vals[k + 2])
                    # h = sigmoid(box_vals[k + 3])
                    w = box_vals[k + 2]
                    h = box_vals[k + 3]
                    cell_topleft_x = col * GRID_WIDTH
                    cell_topleft_y = row * GRID_HEIGHT
                    # cell_center_x = cell_topleft_x + GRID_WIDTH / 2
                    # cell_center_y = cell_topleft_y + GRID_HEIGHT / 2
                    # Unnormalize values
                    # x = int(cell_center_x - (x * (GRID_WIDTH / 2)))
                    x = int(cell_topleft_x + (x * GRID_WIDTH))
                    # y = int(cell_center_y - (y * (GRID_HEIGHT / 2)))
                    y = int(cell_topleft_y + (y * GRID_HEIGHT))
                    w = int(w * IMG_WIDTH)
                    h = int(h * IMG_HEIGHT)
                    box_dict[l].append([x, y, w, h])
    return box_dict

def build_or_load(model_dir=MODEL_DIR, allow_load=True):
    from model.model_fn import build_model
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
        box_dict = {
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
                [404, 285, 65, 237],
                [676, 338, 65, 237]
            ],
            1: [
                [145, 242, 60, 253]
            ]
        }

        self.assertEqual(convert_matrix_to_map(convert_map_to_matrix(box_dict, False)), expected)


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
