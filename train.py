import numpy as np
import tensorflow as tf
from dataset import *
from model.model_fn import *
from constants import *

def train(model):
    train_set, val_set = validation_split(load_data())
    train_data, train_targets = train_set
    # print('TRAIN TYPE: ', train_data.dtype)
    # print('TRAIN TYPEsss: ', train_targets.dtype)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_FILE, monitor='loss', save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10),
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=0)
    ]

    model.fit(train_data, train_targets, batch_size=BATCH_SIZE, epochs=200, callbacks=callbacks)#, validation_data=val_set)

def main():
    # Turn on eager execution for debugging
    tf.enable_eager_execution()

    # model = build_model()
    model = Hairy()
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    model.compile(loss='mse', optimizer=optimizer)
    # model.summary()
    train(model)

if __name__ == '__main__':
    main()
