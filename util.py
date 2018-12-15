import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from dataset import *

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

    ds = HairFollicleDataset('data.hdf5')
    index = 0
    image = np.transpose(ds[index][0], (1, 2, 0))
    boxed_image = draw_boxes(image, boxes[index + 1])
    plt.imshow(image)
    plt.show()
