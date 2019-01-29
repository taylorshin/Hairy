import tensorflow as tf
from dataset import *
from model.model_fn import *
from constants import *

def train(model):
    train_set, val_set = validation_split(load_data())
    train_data, train_targets = train_set

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_FILE, monitor='loss', save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    ]

    model.fit(train_data, train_targets, batch_size=BATCH_SIZE, epochs=200, callbacks=callbacks)

def main():
    model = build_model()
    model.summary()
    train(model)

if __name__ == '__main__':
    main()
