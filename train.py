import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import *
from model import Hairy
from constants import *

criterion = nn.MSELoss()

def plot_graph(training_loss, name):
    plt.clf()
    plt.plot(training_loss)
    plt.savefig(OUT_DIR + '/' + name)

def train(model, device, train_loader, val_loader, optimizer, plot=True):
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
                metrics = train_step(model, optimizer, device, data, total_step)

                total_metrics += metrics
                avg_metrics = total_metrics / step
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
    loss = criterion(outputs.permute(0, 2, 3, 1), labels.float())
    loss.backward()
    optimizer.step()

    return np.array([loss.item()])

# def compute_metrics(model, data, total_step):
#     return False

def main():
    parser = argparse.ArgumentParser(description='Trains model')
    parser.add_argument('--batch-size', default=16, type=int, help='Number of data points per batch')
    parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print('Device: ', device)

    # TODO: Add cuda functionality to Hairy
    model = Hairy()
    model.to(device)
    print(model)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    print('Loading data...')
    train_loader, val_loader = get_tv_loaders('data.hdf5', args.batch_size)
    print('Data loaded.')
    print('Training has started.')
    train(model, device, train_loader, val_loader, optimizer, False)

if __name__ == '__main__':
    main()
