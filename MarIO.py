import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description="hyper parameters for the networks")
parser.add_argument("-kp", "--keep_prob", type=float, default=0.8, help="amount to keep during dropout")
args = parser.parse_args()

OUTPUT_SIZE = 5


def create_graph():
    keep_prob = args.keep_prob
    # Fill in shape later since we'll downsample and resize
    inp = tf.placeholder(tf.float32, shape=None)
    w1 = tf.get_variable(name='W1', shape=[5, 5, 4, 24], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable(name='b1', shape=[24], initializer=tf.zeros_initializer)

    w2 = tf.get_variable(name='W4', shape=[5, 5, 24, 36], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable(name='b4', shape=[36], initializer=tf.zeros_initializer)

    w3 = tf.get_variable(name='W4', shape=[5, 5, 36, 48], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable(name='b4', shape=[48], initializer=tf.zeros_initializer)

    w4 = tf.get_variable(name='W4', shape=[3, 3, 48, 64], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable(name='b4', shape=[64], initializer=tf.zeros_initializer)

    w5 = tf.get_variable(name='W5', shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.get_variable(name='b5', shape=[64], initializer=tf.zeros_initializer)

    w_fc1 = tf.get_variable(name="W_fc1", shape=[None, 1164], initializer=tf.contrib.layers.xavier_initializer())
    b_fc1 = tf.get_variable(name="b_fc1", shape=[1164], initializer=tf.contrib.layers.xavier_initializer())

    w_fc2 = tf.get_variable(name="W_fc2", shape=[1164, 100], initializer=tf.contrib.layers.xavier_initializer())
    b_fc2 = tf.get_variable(name="b_fc2", shape=[100], initializer=tf.contrib.layers.xavier_initializer())

    w_fc3 = tf.get_variable(name="W_fc3", shape=[100, 50], initializer=tf.contrib.layers.xavier_initializer())
    b_fc3 = tf.get_variable(name="b_fc3", shape=[50], initializer=tf.contrib.layers.xavier_initializer())

    w_fc4 = tf.get_variable(name="W_fc4", shape=[50, 10], initializer=tf.contrib.layers.xavier_initializer())
    b_fc4 = tf.get_variable(name="b_fc4", shape=[10], initializer=tf.contrib.layers.xavier_initializer())

    w_fc5 = tf.get_variable(name="W_fc5", shape=[10, OUTPUT_SIZE], initializer=tf.contrib.layers.xavier_initializer())
    b_fc5 = tf.get_variable(name="b_fc5", shape=[OUTPUT_SIZE], initializer=tf.contrib.layers.xavier_initializer())

    conv1 = tf.nn.relu(tf.nn.conv2d(inp, w1, strides=[1, 2, 2, 1], padding='valid') + b1)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w2, strides=[1, 2, 2, 1], padding='valid') + b2)
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, w3, strides=[1, 2, 2, 1], padding='valid') + b3)
    conv4 = tf.nn.relu(tf.nn.conv2d(conv3, w4, strides=[1, 1, 1, 1], padding='valid') + b4)
    conv5 = tf.nn.relu(tf.nn.conv2d(conv4, w5, strides=[1, 1, 1, 1], padding='valid') + b5)
    conv5_flattened = tf.contrib.layers.flatten(conv5)

    fc1 = tf.nn.relu(tf.matmul(conv5_flattened, w_fc1) + b_fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)

    fc2 = tf.nn.relu(tf.matmul(fc1, w_fc2) + b_fc1)
    fc2 = tf.nn.dropout(fc2, keep_prob=keep_prob)

    fc3 = tf.nn.relu(tf.matmul(fc2, w_fc3) + b_fc3)
    fc3 = tf.nn.dropout(fc3, keep_prob=keep_prob)

    fc4 = tf.nn.relu(tf.matmul(fc3, w_fc4) + b_fc4)
    fc4 = tf.nn.dropout(fc4, keep_prob=keep_prob)

    fc5 = tf.nn.relu(tf.matmul(fc4, w_fc5) + b_fc5)

    return inp, fc5






