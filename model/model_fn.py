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

    """
    1st term of loss function: x, y coordinates relative to grid cell
    """
    pred_xy = tf.sigmoid(y_pred[..., :2])
    true_xy = y_true[..., :2]
    xy_term = tf.square(true_xy - pred_xy)
    xy_term = one_obj * (xy_term[..., 0] + xy_term[..., 1])
    xy_term = tf.reduce_sum(xy_term)
    xy_term = LAMBDA_COORD * xy_term
    # print('xy term: ', xy_term)

    """
    2nd term of loss function: width and height of bounding box
    """
    pred_wh = tf.sigmoid(y_pred[..., 2:4])
    pred_wh = tf.sqrt(pred_wh)
    true_wh = y_true[..., 2:4]
    true_wh = tf.sqrt(true_wh)
    wh_term = tf.square(true_wh - pred_wh)
    wh_term = one_obj * (wh_term[..., 0] + wh_term[..., 1])
    wh_term = tf.reduce_sum(wh_term)
    wh_term = LAMBDA_COORD * wh_term
    # print('wh term: ', wh_term)

    """
    3rd and 4th terms of loss function: confidence when there is an object and vice versa
    """
    iou_scores = calculate_iou_scores(y_true[..., :4], tf.sigmoid(y_pred[..., :4]))
    pred_box_c = tf.sigmoid(y_pred[..., 4])
    true_box_c = y_true[..., 4] * iou_scores
    square_diff_c = tf.square(true_box_c - pred_box_c)
    c_term_1 = tf.reduce_sum(one_obj * square_diff_c)
    c_term_2 = LAMBDA_NOOBJ * tf.reduce_sum(one_noobj * square_diff_c)
    c_term = c_term_1 + c_term_2
    # print('c term: ', c_term)

    # Combine all terms of the yolo loss function
    loss = xy_term + wh_term + c_term
    return loss

def build_model():
    """
    YOLO-LITE Architecture
    """
    # Input layer
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Layer 1
    x = layers.Conv2D(filters=16, kernel_size=3)(inputs)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2
    x = layers.Conv2D(filters=32, kernel_size=3)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3
    x = layers.Conv2D(filters=64, kernel_size=3)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 4
    x = layers.Conv2D(filters=128, kernel_size=3)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 5
    x = layers.Conv2D(filters=256, kernel_size=3)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = layers.Conv2D(filters=512, kernel_size=3)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 7
    x = layers.Conv2D(filters=512, kernel_size=3)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # Layer 8
    outputs = layers.Conv2D(filters=T, kernel_size=1)(x)

    # Assemble the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    # optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    optimizer = keras.optimizers.Adam(lr=LEARNING_RATE)
    # model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    model.compile(loss=mse_loss, optimizer=optimizer)

    return model
