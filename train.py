import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from dataset import *
from model import Hairy

def train(model, train_loader, val_loader):
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
                print('.')

def train_step(model, optimizer, X, Y, batch_size=50):
    return True

def main():
    parser = argparse.ArgumentParser(description='Trains model')
    parser.add_argument('--batch-size', default=16, type=int, help='Number of data points per batch')
    args = parser.parse_args()

    # TODO: Add cuda functionality to Hairy
    model = Hairy()

    # TODO: OPTIMIZER

    print('Loading data...')
    train_loader, val_loader = get_tv_loaders('data.hdf5', args.batch_size)
    print('Data loaded.')
    print('Training has started.')
    train(model, train_loader, val_loader)

if __name__ == '__main__':
    main()
