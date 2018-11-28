import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from dataset import *
from model import Hairy
from constants import *

# criterion = nn.MSELoss()

def plot_graph(training_loss, name):
    plt.clf()
    plt.plot(training_loss)
    plt.savefig(OUT_DIR + '/' + name)

def train(args, model, device, train_loader, val_loader, optimizer, plot=True):
    model_save = model
    # Number of training steps per epoch
    epoch = 1
    total_step = 1

    # Keeps track of all losses for each epoch
    train_metrics = []
    val_metrics = []

    # Epoch loop
    while True:
        # Training
        step = 1
        total_metrics = 0

        with tqdm(train_loader) as t:
            t.set_description('Epoch {}'.format(epoch))

            for data in t:
                _, labels = data
                # Baseline MSE for current batch
                base_loss = F.mse_loss(labels, torch.zeros((args.batch_size, S1, S2, T), dtype=torch.double)).item()

                metrics = train_step(model, optimizer, device, data, total_step)

                total_metrics += metrics
                avg_metrics = total_metrics / step
                # t.set_postfix(loss=avg_metrics[0], base=base_loss)
                t.set_postfix(loss=avg_metrics[0])

                step += 1
                total_step += 1

        train_metrics.append(avg_metrics)

        # TODO: validation

        if plot:
            plot_graph([m[0] for m in train_metrics], 'loss.png')

        # Save model
        torch.save(model_save.state_dict(), OUT_DIR + '/model' + '.pt')

        epoch += 1

def train_step(model, optimizer, device, data, total_step):
    model.train()

    # loss, metrics = compute_metrics(model, data, total_step)
    inputs, labels = data
    # Send inputs and targets at every step to GPU if available
    inputs, labels = inputs.to(device), labels.to(device)

    # Zero out gradients
    optimizer.zero_grad()

    # Forward pass -> backward pass -> optimize
    outputs = model(inputs)
    # Transform network outputs
    # outputs = torch.sigmoid(outputs)
    # loss = criterion(outputs.permute(0, 2, 3, 1), labels.float())
    # loss = criterion(outputs, labels.float())
    # YOLO
    # y_true is a DoubleTensor so convert it to FloatTensor to match pred
    loss = yolo_loss(outputs, labels.float())
    loss.backward()
    optimizer.step()

    return np.array([loss.item()])

# def compute_metrics(model, data, total_step):
#     return False

def yolo_loss(y_pred, y_true):
    # 1 when there is object, 0 when there is no object in cell
    one_obj = torch.unsqueeze(y_true[..., 4], 3)
    # 1 when there is no object, 0 when there is object
    one_noobj = 1.0 - one_obj

    # 1st term of loss function: x, y
    pred_xy = torch.sigmoid(y_pred[..., :2])
    true_xy = y_true[..., :2]
    xy_term = torch.pow(true_xy - pred_xy, 2)
    xy_term = one_obj * xy_term
    xy_term = torch.sum(xy_term)
    xy_term = LAMBDA_COORD * xy_term.item()

    # 2nd term of loss function: w, h
    pred_wh = torch.sigmoid(y_pred[..., 2:4])
    pred_wh = torch.sqrt(pred_wh)
    true_wh = y_true[..., 2:4]
    true_wh = torch.sqrt(true_wh)
    wh_term = torch.pow(true_wh - pred_wh, 2)
    wh_term = one_obj * wh_term
    wh_term = torch.sum(wh_term)
    wh_term = LAMBDA_COORD * wh_term.item()

    # 3rd and 4th terms of loss function: confidence
    box_true_xy = y_true[..., :2]
    box_true_wh = y_true[..., 2:4]
    box_true_wh_half = box_true_wh / 2.0
    true_mins = box_true_xy - box_true_wh_half
    true_maxs = box_true_xy + box_true_wh_half

    # Is sigmoid needed?
    box_pred_xy = torch.sigmoid(y_pred[..., :2])
    box_pred_wh = torch.sigmoid(y_pred[..., 2:4])
    box_pred_wh_half = box_pred_wh / 2.0
    pred_mins = box_pred_xy - box_pred_wh_half
    pred_maxs = box_pred_xy + box_pred_wh_half

    intersect_mins = torch.max(pred_mins, true_mins)
    intersect_maxs = torch.min(pred_maxs, true_maxs)
    # TODO: FIX THIS CUDA SHINDIG
    intersect_wh = torch.max(intersect_maxs - intersect_mins, torch.zeros(intersect_mins.size()).cuda())
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = box_true_wh[..., 0] * box_true_wh[..., 1]
    pred_areas = box_pred_wh[..., 0] * box_pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    # Combine first and second half of confidence term
    pred_conf = torch.sigmoid(y_pred[..., 4])
    # TODO: Consolidate the dim of one_obj and one_noobj
    one_obj = torch.squeeze(one_obj)
    one_noobj = torch.squeeze(one_noobj)
    conf_term_1 = torch.sum(one_obj * torch.pow(pred_conf - iou_scores, 2))
    conf_term_2 = LAMBDA_NOOBJ * torch.sum(one_noobj * torch.pow(pred_conf - iou_scores, 2))
    conf_term = conf_term_1 + conf_term_2

    # Combine all terms of the yolo loss function
    loss = xy_term + wh_term + conf_term
    return loss

def main():
    parser = argparse.ArgumentParser(description='Trains model')
    parser.add_argument('--batch-size', default=16, type=int, help='Number of data points per batch')
    parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print('Device: ', device)

    model = Hairy()
    model.to(device)
    print(model)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    print('Loading data...')
    train_loader, val_loader = get_tv_loaders('data.hdf5', args.batch_size)
    # Save test data indices for predict to use
    # for data in val_loader:
    #     print(data)
    print('Data loaded.')
    print('Training has started.')
    train(args, model, device, train_loader, val_loader, optimizer, False)

if __name__ == '__main__':
    main()
