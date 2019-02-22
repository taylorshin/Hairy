import numpy as np
import argparse
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
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

def predict_data_set(model, data, labels):
    """
    Loop through (test) dataset, make bounding box predictions on each image, and save the updated images
    """
    for i in tqdm(range(data.shape[0])):
        og_img = data[i, :, :, 5]
        og_img = np.tile(og_img[:, :, np.newaxis], (1, 1, 3))
        inputs = tf.expand_dims(data[i, :, :, :], 0)
        prediction = model.predict(inputs, steps=1)
        boxes = convert_matrix_to_map(prediction, 0.4)
        box_img = draw_boxes(og_img * 255, boxes[0])
        box_img = np.squeeze(box_img)
        box_img = box_img.astype('uint8')
        save_img = Image.fromarray(box_img, 'RGB')
        save_img.save(PREDICT_DIR + str(i) + '.jpg', 'JPEG')

def predict_data_point(model, data, labels, index, conf_thresh):
    """
    Predict bounding boxes around hair follicles for a single data point at index
    """
    og_img = data[index, :, :, 5]#.astype(int)
    og_img = og_img[:, :, np.newaxis]
    og_img = np.tile(og_img, (1, 1, 3))
    # plt.imshow(og_img)
    # plt.show()

    # Add batch dimension
    inputs = tf.expand_dims(data[index, :, :, :], 0)

    # Feed the input to the model
    prediction = model.predict(inputs, steps=1)

    # Transform network output to obtain bounding box predictions
    # prediction = tf.sigmoid(prediction)

    boxes = convert_matrix_to_map(prediction, conf_thresh)
    print('BOXES: ', boxes)

    box_img = draw_boxes(og_img, boxes[0])
    box_img = np.squeeze(box_img)
    plt.imshow(box_img)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Make predictions from trained model')
    parser.add_argument('--model', default=MODEL_DIR, type=str, help='Path to model file')
    parser.add_argument('--img', default=-1, type=int, help='Image index')
    parser.add_argument('--conf', default=0.5, type=float, help='Confidence threshold')
    args = parser.parse_args()

    # Load the model
    model = build_or_load(args.model)

    data = load_3d_data('data/J_data')
    targets = load_labels('data/labels/image_boxes_J.txt')
    print('DATA: ', data.shape)
    print('TARGETS: ', targets.shape)

    # Predict bounding boxes
    if args.img < 0:
        # Make predictions on all images
        predict_data_set(model, data, targets)
    else:
        # Make prediction on single image
        predict_data_point(model, data, targets, args.img, args.conf)


if __name__ == '__main__':
    main()
