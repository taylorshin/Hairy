import tensorflow as tf
from dataset import *
from model.model_fn import *

def train(model):
    train_set, val_set = validation_split(load_data())
    train_data, train_targets = train_set
    print('data: ', train_data.shape)
    print('targets: ', train_targets.shape)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    ]

    model.fit(train_data, train_targets, batch_size=8, epochs=5, callbacks=callbacks)

def main():
    model = build_model()
    train(model)

if __name__ == '__main__':
    main()
