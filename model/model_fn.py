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

def calculate_iou_scores(true_boxes, pred_boxes):
    x_min = tf.minimum(true_boxes[:, :, :, 0] - (true_boxes[:, :, :, 2] / 2.0), pred_boxes[:, :, :, 0] - (pred_boxes[:, :, :, 2] / 2.0))
    x_max = tf.maximum(true_boxes[:, :, :, 0] + (true_boxes[:, :, :, 2] / 2.0), pred_boxes[:, :, :, 0] + (pred_boxes[:, :, :, 2] / 2.0))
    y_min = tf.minimum(true_boxes[:, :, :, 1] - (true_boxes[:, :, :, 3] / 2.0), pred_boxes[:, :, :, 1] - (pred_boxes[:, :, :, 3] / 2.0))
    y_max = tf.maximum(true_boxes[:, :, :, 1] + (true_boxes[:, :, :, 3] / 2.0), pred_boxes[:, :, :, 1] + (pred_boxes[:, :, :, 3] / 2.0))
    w1 = true_boxes[:, :, :, 2]
    h1 = true_boxes[:, :, :, 3]
    w2 = pred_boxes[:, :, :, 2]
    h2 = pred_boxes[:, :, :, 3]

    w_union = x_max - x_min
    h_union = y_max - y_min
    w_intersect = w1 + w2 - w_union
    h_intersect = h1 + h2 - h_union

    w_intersect = tf.maximum(w_intersect, tf.zeros(tf.shape(w_intersect)))
    h_intersect = tf.maximum(h_intersect, tf.zeros(tf.shape(h_intersect)))

    intersect_area = w_intersect * h_intersect

    true_boxes_area = w1 * h1
    pred_boxes_area = w2 * h2

    iou = intersect_area / (true_boxes_area + pred_boxes_area - intersect_area)

    return iou

def yolo_loss(y_true, y_pred):
    # print('y_true: ', tf.shape(y_true))#, y_true)
    # print('y_pred: ', tf.shape(y_pred))#, y_pred)
    # y_pred = tf.cast(y_pred, tf.float32)
    # y_true = tf.cast(y_true, tf.float32)
    # 1 when there is object, 0 when there is no object in cell
    one_obj = y_true[..., 4]
    # 1 when there is no object, 0 when there is object
    one_noobj = 1.0 - one_obj

    # 1st term of loss function: x, y
    # pred_xy = tf.math.sigmoid(y_pred[..., :2])
    pred_xy = y_pred[..., :2]
    true_xy = y_true[..., :2]
    xy_term = keras.backend.pow(true_xy - pred_xy, 2)
    xy_term = one_obj * (xy_term[..., 0] + xy_term[..., 1])
    xy_term = keras.backend.sum(xy_term)
    xy_term = LAMBDA_COORD * xy_term
    # print('xy term: ', xy_term)

    # 2nd term of loss function: w, h
    # pred_wh = tf.math.sigmoid(y_pred[..., 2:4])
    pred_wh = y_pred[..., 2:4]
    pred_wh = keras.backend.sqrt(pred_wh)
    true_wh = y_true[..., 2:4]
    true_wh = keras.backend.sqrt(true_wh)
    wh_term = keras.backend.pow(true_wh - pred_wh, 2)
    wh_term = one_obj * (wh_term[..., 0] + wh_term[..., 1])
    wh_term = keras.backend.sum(wh_term)
    wh_term = LAMBDA_COORD * wh_term
    # print('wh term: ', wh_term)

    # Temporary MSE loss for confidence
    # pred_c = tf.math.sigmoid(y_pred[..., 4])
    # pred_c = y_pred[..., 4]
    # true_c = y_true[..., 4]
    # conf_term = tf.math.pow(pred_c - true_c, 2)
    # conf_term = tf.math.reduce_sum(conf_term) / tf.cast(tf.shape(true_c)[0], tf.float32)
    pred_c = calculate_iou_scores(y_true[..., :4], y_pred[..., :4])
    true_c = y_true[..., 4]
    square_diff_c = keras.backend.pow(true_c - pred_c, 2)
    c_term_1 = one_obj * square_diff_c
    c_term_1 = keras.backend.sum(c_term_1)
    c_term_2 = one_noobj * square_diff_c
    c_term_2 = LAMBDA_NOOBJ * keras.backend.sum(c_term_2)
    c_term = c_term_1 + c_term_2

    # Combine all terms of the yolo loss function
    loss = xy_term + wh_term + c_term
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
    # x = layers.Dense(4096, activation='softmax')(x)
    x = layers.Dense(4096, activation=None)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    # x = layers.Dense(S1 * S2 * T, activation='sigmoid')(x)
    x = layers.Dense(S1 * S2 * T, activation=None)(x)
    outputs = layers.Reshape((S1, S2, T))(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    # model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    model.compile(loss=yolo_loss, optimizer=optimizer)

    return model
