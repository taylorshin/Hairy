import numpy as np
import argparse
import tensorflow as tf
from dataset import *
from model.model_fn import *
from constants import *
from util import *

def main():
    parser = argparse.ArgumentParser(description='Make predictions from trained model')
    # parser.add_argument('model', help='Path to model file')
    parser.add_argument('--img', default=0, type=int, help='Image index')
    parser.add_argument('--conf', default=0.5, type=float, help='Confidence threshold')
    args = parser.parse_args()

    # print('Model path: {}'.format(args.model))

    model = build_or_load()
    # model.summary()

    val_set, val_set = validation_split(load_2d_data())
    data, targets = val_set
    # data = np.squeeze(data)

    index = args.img
    plt_img = data[index]
    # plt.imshow(plt_img)
    # plt.show()

    # Add batch dimension
    test_img = tf.expand_dims(plt_img, 0)

    prediction = model.predict(test_img, steps=1)
    # Need to permute dims for YOLO v2
    # output = output.permute(0, 2, 3, 1)
    # print('Prediction: ', prediction)

    # Transform network output to obtain bounding box predictions
    prediction = tf.math.sigmoid(prediction)

    sess = tf.Session()
    with sess.as_default():
        boxes = convert_matrix_to_map(prediction.eval(), args.conf)
        print('BOXES: ', boxes)

        box_img = draw_boxes(plt_img, boxes[0])
        box_img = np.squeeze(box_img)
        plt.imshow(box_img)
        plt.show()

if __name__ == '__main__':
    main()
