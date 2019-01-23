import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model():
    model = keras.Sequential([
        layers.Conv2D(3, 32, 7),
        layers.LeakyReLU(),
        layers.MaxPooling2D(pool_size=(2, 2))
    ])

    optimizer = tf.train.AdamOptimizer()

    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

    return model
