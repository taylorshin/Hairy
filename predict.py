import numpy as np
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import *
from model import Hairy
from constants import *

def predict(model, image):
    # model.eval()
    output = model(image)
    return output

def main():
    parser = argparse.ArgumentParser(description='Make predictions from trained model')
    parser.add_argument('model', help='Path to model file')
    args = parser.parse_args()

    print('Model path: {}'.format(args.model))

    model = Hairy()

    if args.model:
        model.load_state_dict(torch.load(args.model))
    else:
        print('WARNING: No model loaded! Please specify model path.')

    # print('Loading data...')
    ds = HairFollicleDataset('data.hdf5')
    index = 0
    plt_image = np.transpose(ds[index][0], (1, 2, 0))
    image = torch.unsqueeze(torch.from_numpy(ds[index][0]), 0)
    print('image size: ', image.size())

    output = predict(model, image).permute(0, 2, 3, 1)
    print('model output: ', output[0, 0, 0], output.size())

    boxes = convert_matrix_to_map_2(output.detach().numpy())
    print('BOXES: ', boxes)
    boxed_image = draw_boxes(plt_image, boxes[index])
    plt.imshow(boxed_image)
    plt.show()

if __name__ == '__main__':
    main()
