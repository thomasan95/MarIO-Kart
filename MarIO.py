import tensorflow as tf
import config
import numpy as np
import argparse
# Load Configs
conf = config.Config()
parser = argparse.ArgumentParser(description="specify task of the network")
parser.add_argument("-t", "-task", type=str, default='train', help="will implement other parameters")
args = parser.parse_args()

'''
class Sample:
    IMG_W = 200
    IMG_H = 66
    IMG_D = 3
'''


def create_graph():
    """
    Instantiate the graph. We create a placeholder to feed into the network which is then
    created to use through tf.Session()
    Structure:
        Conv1 - Kernel (5,5), Stride (2,2) out-depth 24 (200x66x3 -> 98x31x24)
        Conv2 - Kernel (5,5), Stride (2,2) out-depth 36 (98x31x24 -> 47x14x36)
        Conv3 - Kernel (5,5), Stride (2,2) out-depth 48 (47x14x36 -> 22x5x48)
        Conv4 - Kernel (3,3), Stride (1,1) out-depth 64 (22x5x48 -> 20x3x64)
        Conv5 - Kernel (3,3), Stride (1,1) out-depth 64 (20x3x64 -> 18x1x64)
        Fc1 - [1152, 1164]
        Fc2 - [1164, 100]
        Fc3 = [100, 50]
        Fc4 = [50, 10]
        Fc5 (output) = [10, conf.OUTPUT_SIZE]
    The network along with all important nodes are then then returned so they can then be used to train
    the graph
    :return: graph, input_placeholder, max_actions placeholder, optimal_action, out, action, loss, optimizer
    :rtype: tf.Graph(), tf.placeholder(), tf.placeholder(), tf.placeholder(), tf.Tensor, tf.Tensor, loss, tf optimizer
    """
    graph = tf.Graph()
    with graph.as_default():
        keep_prob = conf.keep_prob
        # Fill in shape later since we'll downsample and resize
        with tf.name_scope("Input"):
            # Input
            inp = tf.placeholder(tf.float32, shape=[None, conf.img_h, conf.img_w, conf.img_d], name="input")
            # For storing what action to take output by the network
            max_action = tf.placeholder(tf.float32, [None, conf.OUTPUT_SIZE], name="max_action")
            # Placeholder for storing optimal action (labels)
            optimal_action = tf.placeholder(tf.float32, [None], name="optimal_action")

        with tf.name_scope("Kernels_and_Bias"):
            w1 = tf.get_variable(name='W1', shape=[5, 5, 3, 24], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable(name='b1', shape=[24], initializer=tf.zeros_initializer)

            w2 = tf.get_variable(name='W2', shape=[5, 5, 24, 36], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable(name='b2', shape=[36], initializer=tf.zeros_initializer)

            w3 = tf.get_variable(name='W3', shape=[5, 5, 36, 48], initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.get_variable(name='b3', shape=[48], initializer=tf.zeros_initializer)

            w4 = tf.get_variable(name='W4', shape=[3, 3, 48, 64], initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.get_variable(name='b4', shape=[64], initializer=tf.zeros_initializer)

            w5 = tf.get_variable(name='W5', shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.get_variable(name='b5', shape=[64], initializer=tf.zeros_initializer)

            w_fc1 = tf.get_variable(name="W_fc1", shape=[1152, 1164],
                                    initializer=tf.contrib.layers.xavier_initializer())
            b_fc1 = tf.get_variable(name="b_fc1", shape=[1164], initializer=tf.contrib.layers.xavier_initializer())

            w_fc2 = tf.get_variable(name="W_fc2", shape=[1164, 100], initializer=tf.contrib.layers.xavier_initializer())
            b_fc2 = tf.get_variable(name="b_fc2", shape=[100], initializer=tf.contrib.layers.xavier_initializer())

            w_fc3 = tf.get_variable(name="W_fc3", shape=[100, 50], initializer=tf.contrib.layers.xavier_initializer())
            b_fc3 = tf.get_variable(name="b_fc3", shape=[50], initializer=tf.contrib.layers.xavier_initializer())

            w_fc4 = tf.get_variable(name="W_fc4", shape=[50, 10], initializer=tf.contrib.layers.xavier_initializer())
            b_fc4 = tf.get_variable(name="b_fc4", shape=[10], initializer=tf.contrib.layers.xavier_initializer())

            w_fc5 = tf.get_variable(name="W_fc5", shape=[10, conf.OUTPUT_SIZE],
                                    initializer=tf.contrib.layers.xavier_initializer())
            b_fc5 = tf.get_variable(name="b_fc5", shape=[conf.OUTPUT_SIZE],
                                    initializer=tf.contrib.layers.xavier_initializer())
        with tf.name_scope("Batch_Norm"):
            inp_batchnorm = tf.contrib.layers.batch_norm(inp, center=True, scale=True, is_training=True)
        with tf.name_scope("Conv_Layers"):
            conv1 = tf.nn.relu(tf.nn.conv2d(inp_batchnorm, w1, strides=[1, 2, 2, 1], padding='VALID') + b1)
            conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w2, strides=[1, 2, 2, 1], padding='VALID') + b2)
            conv3 = tf.nn.relu(tf.nn.conv2d(conv2, w3, strides=[1, 2, 2, 1], padding='VALID') + b3)
            conv4 = tf.nn.relu(tf.nn.conv2d(conv3, w4, strides=[1, 1, 1, 1], padding='VALID') + b4)
            conv5 = tf.nn.relu(tf.nn.conv2d(conv4, w5, strides=[1, 1, 1, 1], padding='VALID') + b5)
        with tf.name_scope("Dense_Layers"):
            conv5_flattened = tf.contrib.layers.flatten(conv5)
            fc1 = tf.nn.relu(tf.matmul(conv5_flattened, w_fc1) + b_fc1)
            fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)
            fc2 = tf.nn.relu(tf.matmul(fc1, w_fc2) + b_fc2)
            fc2 = tf.nn.dropout(fc2, keep_prob=keep_prob)
            fc3 = tf.nn.relu(tf.matmul(fc2, w_fc3) + b_fc3)
            fc3 = tf.nn.dropout(fc3, keep_prob=keep_prob)
            fc4 = tf.nn.relu(tf.matmul(fc3, w_fc4) + b_fc4)
            fc4 = tf.nn.dropout(fc4, keep_prob=keep_prob)
        with tf.name_scope("Predictions"):
            out = tf.nn.relu(tf.matmul(fc4, w_fc5) + b_fc5, name="out")
            action = tf.reduce_sum(tf.multiply(out, max_action), axis=1)
            tf.summary.histogram('predictions', action)
        with tf.name_scope("Loss"):
            loss = tf.reduce_mean(tf.square(action - optimal_action), name="sse_loss")
            tf.summary.scalar('loss', loss)

        with tf.name_scope("Optimizer"):
            global_step = tf.get_variable(name="global_step", shape=[], trainable=False,
                                          initializer=tf.zeros_initializer())
            lr = tf.train.exponential_decay(conf.learning_rate,
                                            global_step,
                                            conf.decay_steps,
                                            conf.anneal_factor,
                                            staircase=True)
            optimizer = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step, name="optim")
    print("Done creating graph")
    return graph, inp, max_action, optimal_action, out, action, loss, optimizer


def train_graph(graph, inp, max_action, optimal_action, out, action, loss, optimizer):
    """
    Main training loop to train the graph
    :param graph:
    :param inp:
    :param max_action:
    :param optimal_action:
    :param out:
    :param action:
    :param loss:
    :param optimizer:
    :return:
    """
    keep_training = True
    with tf.Session(graph=graph) as sess:
        input_tensor = np.ones((conf.img_h, conf.img_w, conf.img_d))
        inp_t = np.stack((input_tensor, input_tensor, input_tensor, input_tensor), axis=0)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(conf.sum_dir + './train/', sess.graph)
        train_iter = 1
        while keep_training:
            out_t = sess.run(out, feed_dict={inp: inp_t})
            print(type(out_t))
            train_iter += 1
            if train_iter % conf.save_freq == 0:
                saver.save(sess, conf.save_dir + conf.save_name, global_step=train_iter)
            break
        train_writer.close()


def main():
    graph, inp, max_action, optimal_action, out, action, loss, optimizer = create_graph()

    if conf.is_training:
        train_graph(graph, inp, max_action, optimal_action, out, action, loss, optimizer)


if __name__ == "__main__":
    main()






