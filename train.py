import argparse
import numpy as np
import tensorflow as tf
from dataset import *
from model.model_fn import *
from constants import *

def train(model):
    # train_data = load_3d_data('data/G_data')
    # train_targets = load_labels('data/labels/image_boxes_G.txt')
    data_paths = ['data/G_data', 'data/H_data', 'data/I_data']
    label_paths = ['data/labels/image_boxes_G.txt', 'data/labels/image_boxes_H.txt', 'data/labels/image_boxes_I.txt']
    train_data, train_targets = load_train_set(data_paths, label_paths)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_DIR, monitor='loss', save_best_only=True, save_weights_only=True),
        # tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10),
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=0)
    ]

    model.fit(train_data, train_targets, batch_size=BATCH_SIZE, epochs=2000, callbacks=callbacks)#, validation_data=val_set)

def main():
    parser = argparse.ArgumentParser(description='Trains the model')
    parser.add_argument('--debug', default=False, type=bool, help='Debug mode')
    args = parser.parse_args()

    if args.debug:
        # Turn on eager execution for debugging
        tf.enable_eager_execution()

    model = build_model()
    model.summary()
    train(model)


if __name__ == '__main__':
    main()
