import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from constants import *

def mse_loss(y_true, y_pred):
    # print('y_true: ', y_true, tf.shape(y_true))
    # print('y_pred: ', y_pred, tf.shape(y_pred))
    mse = tf.reduce_mean(tf.square(y_pred - y_true))
    # print('MSE: ', mse)
    return mse

def yolo_loss(y_true, y_pred):
    # print('y_true: ', y_true, tf.shape(y_true))
    # print('y_pred: ', y_pred, tf.shape(y_pred))
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    # 1 when there is object, 0 when there is no object in cell
    one_obj = y_true[..., 4]
    # 1 when there is no object, 0 when there is object
    one_noobj = 1.0 - one_obj

    # 1st term of loss function: x, y
    pred_xy = tf.math.sigmoid(y_pred[..., :2])
    true_xy = y_true[..., :2]
    xy_term = tf.math.pow(true_xy - pred_xy, 2)
    xy_term = one_obj * (xy_term[..., 0] + xy_term[..., 1])
    xy_term = tf.math.reduce_sum(xy_term)
    xy_term = LAMBDA_COORD * xy_term
    # print('xy term: ', xy_term)

    # 2nd term of loss function: w, h
    pred_wh = tf.math.sigmoid(y_pred[..., 2:4])
    pred_wh = tf.math.sqrt(pred_wh)
    true_wh = y_true[..., 2:4]
    true_wh = tf.math.sqrt(true_wh)
    wh_term = tf.math.pow(true_wh - pred_wh, 2)
    wh_term = one_obj * (wh_term[..., 0] + wh_term[..., 1])
    wh_term = tf.math.reduce_sum(wh_term)
    wh_term = LAMBDA_COORD * wh_term

    # Temporary MSE loss for confidence
    pred_c = tf.math.sigmoid(y_pred[..., 4])
    true_c = y_true[..., 4]
    conf_term = tf.math.pow(pred_c - true_c, 2)
    tf.print(true_c.shape)
    conf_term = tf.math.reduce_sum(conf_term) / tf.cast(tf.shape(true_c)[0], tf.float32)

    # Combine all terms of the yolo loss function
    # print('xy: ', xy_term)
    # print('wh: ', wh_term)
    # print('conf: ', conf_term)
    loss = xy_term + wh_term + conf_term
    return loss

def build_model():
    # Input layer
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Layer 1
    x = layers.Conv2D(filters=32, kernel_size=7)(inputs)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2
    x = layers.Conv2D(filters=32, kernel_size=5)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3
    x = layers.Conv2D(filters=32, kernel_size=5)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 4
    x = layers.Conv2D(filters=32, kernel_size=5)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 5
    x = layers.Conv2D(filters=32, kernel_size=5)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = layers.Conv2D(filters=32, kernel_size=3)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Classification layers
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='softmax')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dense(S1 * S2 * T, activation='sigmoid')(x)
    outputs = layers.Reshape((S1, S2, T))(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    # model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    model.compile(loss=mse_loss, optimizer=optimizer)

    return model

class Hairy(keras.Model):
    
    def __init__(self):
        super(Hairy, self).__init__(name='Hairy')
        self.conv1 = layers.Conv2D(32, 7)
        self.conv2 = layers.Conv2D(32, 5)
        self.conv3 = layers.Conv2D(32, 5)
        self.conv4 = layers.Conv2D(32, 5)
        self.conv5 = layers.Conv2D(32, 5)
        self.conv6 = layers.Conv2D(32, 3)
        self.lrelu = layers.LeakyReLU(alpha=0.1)
        self.pool = layers.MaxPooling2D(pool_size=(2, 2))
        self.dense1 = layers.Dense(4096, activation='softmax')
        self.dense2 = layers.Dense(S1 * S2 * T, activation='softmax')

    def call(self, inputs):
        print('before conv layers 1: ', inputs)
        # Conv layers 1
        x = self.conv1(inputs)
        x = self.lrelu(x)
        x = self.pool(x)
        print('after conv layers 1: ', x)

        # Conv layers 2
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.pool(x)

        # Conv layers 3
        x = self.conv3(x)
        x = self.lrelu(x)
        x = self.pool(x)

        # Conv layers 4
        x = self.conv4(x)
        x = self.lrelu(x)
        x = self.pool(x)

        # Conv layers 5
        x = self.conv5(x)
        x = self.lrelu(x)
        x = self.pool(x)

        # Conv layers 6
        x = self.conv6(x)
        x = self.lrelu(x)
        x = self.pool(x)

        # Fully connected layers
        x = layers.Flatten()(x)
        x = self.dense1(x)
        x = self.lrelu(x)
        x = self.dense2(x)
        outputs = layers.Reshape((S1, S2, T))(x)

        return outputs
