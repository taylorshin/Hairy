import numpy as np
import argparse
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
from dataset import *
from model import *
from constants import *
from util import *

def predict_data_set(model, save_path, data, label_dict, conf_thresh):
    """
    Loop through (test) dataset, make bounding box predictions on each image, and save the updated images
    """
    for i in tqdm(range(data.shape[0])):
        # Original (O.G.) image
        og_img = np.expand_dims(data[i, :, :, CONTEXT_FRAMES], axis=2)
        og_img = np.tile(og_img, (1, 1, 3))

        # Make prediction
        inputs = tf.expand_dims(data[i, :, :, :], axis=0)
        prediction = model.predict(inputs, steps=1)
        boxes = convert_matrix_to_dict(prediction, conf_thresh)

        # Plot predicted boxes
        pred_img = draw_boxes(og_img * 255, boxes[0])
        pred_img = np.squeeze(pred_img)

        # Plot true boxes
        # Replace class values which can be 0, 1, or 2 for true labels with 1 indicating 100% confidence
        og_labels = [(np.array(boxes[:-1]) / DOWNSCALE_FACTOR).tolist() + [1] for boxes in label_dict[i]]
        true_img = draw_boxes(og_img * 255, og_labels)
        true_img = np.squeeze(true_img)

        # Plot and save both predicted and true images side by side
        compare_img = np.vstack((pred_img, true_img))
        compare_img = compare_img.astype('uint8')
        save_img = Image.fromarray(compare_img, 'RGB')
        save_img.save(save_path + str(i) + '.png', 'PNG')

def predict_data_point(model, data, label_dict, index, conf_thresh):
    """
    Predict bounding boxes around hair follicles for a single data point at index
    """
    og_img = np.expand_dims(data[index, :, :, CONTEXT_FRAMES], axis=2)
    og_img = np.tile(og_img, (1, 1, 3))

    # Add batch dimension
    inputs = tf.expand_dims(data[index, :, :, :], axis=0)

    # Feed the input to the model
    prediction = model.predict(inputs, steps=1)

    # Transform network output ONLY for YOLO loss!!!
    # prediction = sigmoid(prediction)

    # Convert predictions from matrix form to dictionary
    boxes = convert_matrix_to_dict(prediction, conf_thresh)
    print('Predicted boxes: ', boxes)

    # fig, axes = plt.subplots(nrows=2, ncols=1)

    # Plot predicted boxes
    pred_img = draw_boxes(og_img, boxes[0])
    pred_img = np.squeeze(pred_img)
    # axes[0].imshow(pred_img)

    # Plot true bounding boxes
    # Replace class values which can be 0, 1, or 2 for true labels with 1 indicating 100% confidence
    og_labels = [(np.array(boxes[:-1]) / DOWNSCALE_FACTOR).tolist() + [1] for boxes in label_dict[index]]
    true_img = draw_boxes(og_img, og_labels)
    true_img = np.squeeze(true_img)
    # axes[1].imshow(true_img)

    # Plot predicted and true boxes on same plot for side by side comparison
    compare_img = np.vstack((pred_img, true_img))
    plt.imshow(compare_img)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Make predictions from trained model')
    parser.add_argument('--model', default=MODEL_DIR, type=str, help='Path to model file')
    parser.add_argument('--save', default=PREDICT_DIR, type=str, help='Directory where predicted images will be saved')
    parser.add_argument('--data', default='G', type=str, help='Set of data to make predictions on (H, G, I, J)')
    parser.add_argument('--img', default=-1, type=int, help='Image index')
    parser.add_argument('--conf', default=0.5, type=float, help='Confidence threshold')
    args = parser.parse_args()

    # Load the model
    model = build_or_load(args.model)
    # Load original data
    data = load_3d_data('data/' + args.data + '_data')
    # Load original labels
    label_dict = load_label_dict('data/labels/image_boxes_' + args.data + '.txt')
    # Modify dictionary keys from strings to ints
    label_dict = { int((int(key) / 20) - 1) : value for key, value in label_dict.items() }

    # Predict bounding boxes
    if args.img < 0:
        # Make predictions on all images
        predict_data_set(model, args.save, data, label_dict, args.conf)
    else:
        # Make prediction on single image
        predict_data_point(model, data, label_dict, args.img, args.conf)


if __name__ == '__main__':
    main()
