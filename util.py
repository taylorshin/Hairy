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
        print(box)
        x, y, w, h = box
        x_min = int(x - w / 2.0)
        y_min = int(y - h / 2.0)
        x_max = int(x + w / 2.0)
        y_max = int(y + h / 2.0)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (1, 0, 0), 3)
    return image

def bb_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Area of intersection rectangle
    intersect_area = (x2 - x1) * (y2 - y1)

    # Compute area of prediction and ground truth boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute intersection over union
    iou = intersect_area / float(box1_area + box2_area - intersect_area)

    return iou

if __name__ == '__main__':
    tfile = open('image_boxes.txt', 'r')
    content = tfile.read()
    boxes = eval(content)

    ds = HairFollicleDataset('data.hdf5')
    index = 0
    image = np.transpose(ds[index][0], (1, 2, 0))
    boxed_image = draw_boxes(image, boxes[index + 1])
    plt.imshow(image)
    plt.show()
