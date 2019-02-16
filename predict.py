import numpy as np
import argparse
import tensorflow as tf
from dataset import *
from model.model_fn import *
from constants import *
from util import *

def get_mean_iou(predictions, labels):
    """
    Loop through test set and calculate the mean IOU score
    """
    assert len(predictions) == len(labels)
    # for p, l in zip(predictions, labels):

def main():
    parser = argparse.ArgumentParser(description='Make predictions from trained model')
    parser.add_argument('--img', default=0, type=int, help='Image index')
    parser.add_argument('--conf', default=0.5, type=float, help='Confidence threshold')
    args = parser.parse_args()

    model = build_or_load()

    # val_set, val_set = validation_split(load_2d_data())
    # data, targets = val_set
    # data = np.squeeze(data)

    data = load_3d_data('data/H_data')
    targets = load_labels('data/labels/image_boxes_H.txt')
    print('DATA: ', data.shape)
    print('TARGETS: ', targets.shape)

    index = 1 #args.img
    plt_img = data[index, :, :, 5]#.astype(int)
    plt_img = plt_img[:, :, np.newaxis]
    plt_img = np.tile(plt_img, (1, 1, 3))
    # plt_img = np.transpose(plt_img, (2, 0, 1))
    # print('plt img: ', plt_img)
    # plt.imshow(plt_img)
    # plt.show()

    # Add batch dimension
    # test_img = tf.expand_dims(plt_img, 0)
    test_img = tf.expand_dims(data[index, :, :, :], 0)

    prediction = model.predict(test_img, steps=1)
    # Need to permute dims for YOLO v2
    # output = output.permute(0, 2, 3, 1)
    # print('Prediction: ', prediction)

    # Transform network output to obtain bounding box predictions
    # prediction = tf.math.sigmoid(prediction)

    sess = tf.Session()
    with sess.as_default():
        # print('prediction: ', tf.shape(prediction), prediction)
        boxes = convert_matrix_to_map(prediction, args.conf)
        print('BOXES: ', boxes)

        box_img = draw_boxes(plt_img, boxes[0])
        box_img = np.squeeze(box_img)
        plt.imshow(box_img)
        plt.show()

if __name__ == '__main__':
    main()
