import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from dataset import *

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

if __name__ == '__main__':
    tfile = open('image_boxes.txt', 'r')
    content = tfile.read()
    boxes = eval(content)

    ds = HairFollicleDataset('data.hdf5')
    index = 0
    image = np.transpose(ds[index][0], (1, 2, 0))
    boxed_image = draw_boxes(image, boxes[index + 1])
    plt.imshow(boxed_image)
    plt.show()
