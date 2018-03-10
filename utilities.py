import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize
import config
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from inputs import get_gamepad
import threading
import sys
conf = config.Config()


def get_batches(x_train, y_train, batch_i, last_batch, remaining_batch):
    """
    Grab batches for supervised training
    :param x_train: x training data
    :type x_train: numpy.ndarray
    :param y_train: labels for training data
    :type y_train: numpy.ndarray
    :param batch_i: current batch for the data
    :type batch_i: int
    :param last_batch: boolean to grab last data or not
    :type last_batch: bool
    :param remaining_batch: how much of the data is left over for this batch
    :type remaining_batch: int
    :returns: batch_x, batch_y
    """
    if last_batch:
        return x_train[-remaining_batch:], y_train[-remaining_batch:]
    else:
        start_idx = batch_i * conf.batch_size
        end_idx = batch_i * conf.batch_size + conf.batch_size
        batch_x = x_train[start_idx:end_idx]
        batch_y = y_train[start_idx:end_idx]
        yield batch_x, batch_y


def resize_img(img):
    """
    Image to pass in
    :param img: numpy array image
    :return: resized image
    """
    im = resize(img, (conf.img_h, conf.img_w, conf.img_d))
    im_arr = im.reshape((conf.img_h, conf.img_w, conf.img_d))
    return im_arr