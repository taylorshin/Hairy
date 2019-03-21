import os
import argparse
import numpy as np
import tensorflow as tf
from dataset import *
from model import *
from constants import *

def train(model, batch_size, model_dir=MODEL_DIR):
    train_data_dirs = ['data/G_data']#, 'data/H_data']
    train_label_files = ['data/labels/image_boxes_G.txt']#, 'data/labels/image_boxes_H.txt']
    val_data_dirs = ['data/I_data']
    val_label_files = ['data/labels/image_boxes_I.txt']
    train_generator = DataGenerator(train_data_dirs, train_label_files, batch_size, enable_data_aug=True)
    # val_generator = DataGenerator(val_data_dirs, val_label_files, batch_size, enable_data_aug=False)

    # Check if data looks correct
    # verify_data_generator(train_generator)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_dir, monitor='loss', save_best_only=True, save_weights_only=True),
        # tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, verbose=1),
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=10, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=0)
    ]

    # TODO: Try the use_multiprocessing parameter
    return model.fit_generator(
                                generator=train_generator,
                                epochs=100,
                                callbacks=callbacks
                                #validation_data=val_generator
                            )

def main():
    parser = argparse.ArgumentParser(description='Trains the model')
    parser.add_argument('--debug', default=False, type=bool, help='Debug mode')
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='Size of training batch')
    args = parser.parse_args()

    if args.debug:
        # Turn on eager execution for debugging
        tf.enable_eager_execution()

    model = build_model()
    model.summary()
    history = train(model, args.batch_size)
    
    # lrs = np.arange(0.00001, 0.0001, 0.00002)
    # losses = []
    # for lr in lrs:
    #     model = build_model(lr)
    #     model_dir = os.path.join('out', 'model_' + str(lr) + '.h5')
    #     history = train(model, model_dir)
    #     losses.append(history.history['loss'][-1])
    #     print('Losses: ', losses)

    #     train_loss = history.history['loss']
    #     epochs = range(len(train_loss))
    #     plt.plot(epochs, train_loss, label='Training Loss')
    #     # plt.plot(epochs, val_loss, label='Validation Loss')
    #     plt.title('Training Loss')
    #     # plt.legend()
    #     plot_file = os.path.join('out', 'loss_' + str(lr) + '.png')
    #     plt.savefig(plot_file)

    # smallest_loss = np.min(losses)
    # print('Smallest loss: ', smallest_loss)
    # index = np.argmin(losses)
    # print('Learning rate: ', lrs[index])
    
    ### Plot training and validation loss over epochs ###
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(train_loss))
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(PLOT_FILE)


if __name__ == '__main__':
    main()
