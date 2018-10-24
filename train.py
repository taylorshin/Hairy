import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
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

def train(model, train_loader, val_loader, optimizer, plot=True):
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
                metrics = train_step(model, optimizer, data, total_step)

                total_metrics += metrics
                avg_metrics = total_metrics / step
                # t.set_postfix(loss=avg_metrics[0])

                step += 1
                total_step += 1

        train_metrics.append(avg_metrics)

        # TODO: validation

        if plot:
            plot_graph([m[0] for m in train_metrics], 'loss.png')

        # Save model
        torch.save(model_save.state_dict(), OUT_DIR + '/model' + '.pt')

        epoch += 1

def train_step(model, optimizer, data, total_step):
    model.train()

    loss, metrics = compute_metrics(model, data, total_step)

    optimizer.zero_grad()
    optimizer.backward(loss)
    optimizer.step()

    return metrics

def compute_metrics(model, data, total_step):
    return 0, 0

def main():
    parser = argparse.ArgumentParser(description='Trains model')
    parser.add_argument('--batch-size', default=16, type=int, help='Number of data points per batch')
    args = parser.parse_args()

    # TODO: Add cuda functionality to Hairy
    model = Hairy()

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    print('Loading data...')
    train_loader, val_loader = get_tv_loaders('data.hdf5', args.batch_size)
    print('Data loaded.')
    print('Training has started.')
    train(model, train_loader, val_loader, optimizer)

if __name__ == '__main__':
    main()