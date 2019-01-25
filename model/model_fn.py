import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from constants import *

def build_model():
    # model = keras.Sequential([
    #     layers.Conv2D(3, 32, 7),
    #     layers.LeakyReLU(),
    #     layers.MaxPooling2D(pool_size=(2, 2)),
    #     layers.Conv2D(32, 32, 5),
    #     layers.LeakyReLU(),
    #     layers.MaxPooling2D(pool_size=(2, 2)),
    #     layers.Conv2D(32, 32, 5),
    #     layers.LeakyReLU(),
    #     layers.MaxPooling2D(pool_size=(2, 2)),
    #     layers.Conv2D(32, 32, 5),
    #     layers.LeakyReLU(),
    #     layers.MaxPooling2D(pool_size=(2, 2)),
    #     layers.Conv2D(32, 32, 5),
    #     layers.LeakyReLU(),
    #     layers.MaxPooling2D(pool_size=(2, 2)),
    #     layers.Conv2D(32, 32, 3),
    #     layers.LeakyReLU(),
    #     layers.MaxPooling2D(pool_size=(2, 2))
    # ])
    # model.summary()

    # Input layer
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Layer 1
    x = layers.Conv2D(3, 32, 7)(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2
    x = layers.Conv2D(32, 32, 5)(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3
    x = layers.Conv2D(3, 32, 5)(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 4
    x = layers.Conv2D(3, 32, 5)(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 5
    x = layers.Conv2D(3, 32, 5)(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = layers.Conv2D(3, 32, 3)(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # print('before: ', x)
    # x = layers.Flatten()(x)
    # print('after: ', x)

    # Classification layers
    x = layers.Reshape((-1, 32 * 7 * 12))(x)
    x = layers.Dense(4096, activation='softmax')(x)
    x = layers.Dense(S1 * S2 * T, activation='softmax')(x)
    outputs = layers.Reshape((S1, S2, T))(x)
    print('outputs: ', outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.train.AdamOptimizer()
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

    return model
