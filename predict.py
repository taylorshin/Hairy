import numpy as np
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import *
from model import Hairy
from constants import *
from util import *

def predict(model, image):
    model.eval()
    output = model(image)
    return output

def main():
    parser = argparse.ArgumentParser(description='Make predictions from trained model')
    parser.add_argument('model', help='Path to model file')
    args = parser.parse_args()

    print('Model path: {}'.format(args.model))

    model = Hairy()
    # model.eval()

    if args.model:
        model.load_state_dict(torch.load(args.model))
    else:
        print('WARNING: No model loaded! Please specify model path.')

    # print('Loading data...')
    ds = HairFollicleDataset('data.hdf5')
    index = 0
    data_point = ds[index][0]
    plt_image = np.transpose(data_point, (1, 2, 0))
    # plt.imshow(plt_image)
    # plt.show()
    image = torch.unsqueeze(torch.from_numpy(data_point), 0)

    output = predict(model, image).permute(0, 2, 3, 1)
    print('model output: ', output, output.size())

    # Transform network output to obtain bounding box predictions
    # transformed_output = transform_network_output(output.detach().numpy())
    # print('transform output: ', transformed_output)
    # boxes = convert_matrix_to_map(transformed_output)
    output = torch.sigmoid(output)
    boxes = convert_matrix_to_map(output.detach().numpy())
    print('BOXES: ', boxes)

    # IOU
    tfile = open('image_boxes.txt', 'r')
    content = tfile.read()
    box_dict = eval(content)
    # ground_truth_box = (box_dict[1])[0]
    # predicted_box = (boxes[0])[0]
    # first_box_iou = bb_iou(predicted_box, ground_truth_box)
    # print('IOU: ', first_box_iou)

    boxed_image = draw_boxes(plt_image, boxes[0])
    plt.imshow(boxed_image)
    plt.show()

if __name__ == '__main__':
    main()
