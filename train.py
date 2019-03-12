import argparse
import numpy as np
import tensorflow as tf
from dataset import *
from model.model_fn import *
from constants import *

def train(model):
    data_dirs = ['data/G_data', 'data/H_data', 'data/I_data']
    label_files = ['data/labels/image_boxes_G.txt', 'data/labels/image_boxes_H.txt', 'data/labels/image_boxes_I.txt']
    train_generator = DataGenerator(data_dirs, label_files, enable_data_aug=True)

    # Check if data looks correct
    # verify_data_generator(train_generator)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_DIR, monitor='loss', save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, verbose=1),
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.6, patience=5, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=0)
    ]

    # TODO: Try the use_multiprocessing parameter
    return model.fit_generator(
                                generator=train_generator,
                                epochs=1000,
                                callbacks=callbacks
                            )

def main():
    parser = argparse.ArgumentParser(description='Trains the model')
    parser.add_argument('--debug', default=False, type=bool, help='Debug mode')
    args = parser.parse_args()

    if args.debug:
        # Turn on eager execution for debugging
        tf.enable_eager_execution()

    model = build_model()
    model.summary()
    history = train(model)

    ### Plot training and validation loss over epochs ###
    train_loss = history.history['loss']
    # val_loss = history.history['val_loss']
    epochs = range(len(train_loss))
    plt.plot(epochs, train_loss, label='Training Loss')
    # plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training Loss')
    # plt.legend()
    plt.savefig(PLOT_FILE)


if __name__ == '__main__':
    main()
