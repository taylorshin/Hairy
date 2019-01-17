import cv2
import numpy as np
import matplotlib.pyplot as plt
from constants import *
# from dataset import *

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def draw_boxes(image, boxes):
    image_w, image_h, _ = image.shape
    for box in boxes:
        x, y, w, h = box
        x_min = int(x - w / 2.0)
        y_min = int(y - h / 2.0)
        x_max = int(x + w / 2.0)
        y_max = int(y + h / 2.0)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (1, 0, 0), 3)
    return image

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

def convert_map_to_matrix(box_dict):
    """
    Takes in bounding boxes (dict) extracted from processed data and converts them into a label matrix.
    """
    n = len(box_dict)
    labels = np.zeros((n, S1, S2, T))
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
        # TODO: Remove this special code for missing data in old dataset
        i += 1
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

        max_c = np.max(label[:, :, 4::5])
        min_c = np.min(label[:, :, 4::5])
        print('MAX C: {}'.format(max_c))
        print('MIN C: {}'.format(min_c))

        for row in range(num_grid_rows):
            for col in range(num_grid_cols):
                box_vals = label[row, col]
                for k in range(0, T, 5):
                    c = (box_vals[k + 4])
                    # Skip grid if conf prob is less than the threshold
                    if c <= conf_thresh:
                        continue
                    x = (box_vals[k])
                    y = (box_vals[k + 1])
                    # Width and height values are sometimes negative because of leaky relu
                    w = (box_vals[k + 2])
                    print('W: ', w)
                    h = (box_vals[k + 3])
                    print('H: ', h)
                    cell_topleft_x = col * GRID_WIDTH
                    cell_topleft_y = row * GRID_HEIGHT
                    # cell_center_x = cell_topleft_x + GRID_WIDTH / 2
                    # cell_center_y = cell_topleft_y + GRID_HEIGHT / 2
                    # Unnormalize values
                    # x = int(cell_center_x - (x * (GRID_WIDTH / 2)))
                    x = int(cell_topleft_x + x * GRID_WIDTH)
                    # y = int(cell_center_y - (y * (GRID_HEIGHT / 2)))
                    y = int(cell_topleft_y + y * GRID_HEIGHT)
                    w = int(w * IMG_WIDTH)
                    h = int(h * IMG_HEIGHT)
                    box_dict[l].append([x, y, w, h])
    return box_dict

if __name__ == '__main__':
    tfile = open('image_boxes.txt', 'r')
    content = tfile.read()
    boxes = eval(content)

    # IOU Test
    box1 = [5, 5, 4, 4]
    box2 = [7, 7, 4, 4]
    # IOU should be ~0.1429
    iou = bb_iou(box1, box2)
    print('IOU: ', iou)

    # ds = HairFollicleDataset('data.hdf5')
    # index = 0
    # image = np.transpose(ds[index][0], (1, 2, 0))
    # boxed_image = draw_boxes(image, boxes[index + 1])
    # plt.imshow(image)
    # plt.show()
