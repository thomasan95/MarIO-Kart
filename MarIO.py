import tensorflow as tf
import config
import numpy as np
import argparse
import random
import utilities as utils
import gym
print("Using TensorFlow version: " + str(tf.__version__))
# Load Configs
conf = config.Config()
parser = argparse.ArgumentParser(description="specify task of the network")
parser.add_argument("-t", "-task", type=str, default='train', help="will implement other parameters")
args = parser.parse_args()
env = gym.make('Acrobot-v1')


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
    keep_prob = conf.keep_prob
    # Fill in shape later since we'll downsample and resize
    with tf.variable_scope("actor"):
        # actor = tf.Graph()
        # with actor.as_graph_def():
        with tf.variable_scope("actor_input"):
            state_inp = tf.placeholder(tf.float32, shape=conf.inp_shape, name="actor_state_input")
            actor_action = tf.placeholder(tf.float32, shape=env.action_space.n, name="actor_action_ph")
        with tf.variable_scope("actor_kernels_weights"):
            a_w1 = tf.get_variable(name="act_W1", shape=[5, 5, 3, 24],
                                   initializer=tf.contrib.layers.xavier_initializer())
            a_b1 = tf.get_variable(name="act_b1", shape=[24], initializer=tf.zeros_initializer)
            a_w2 = tf.get_variable(name='act_W2', shape=[5, 5, 24, 36],
                                   initializer=tf.contrib.layers.xavier_initializer())
            a_b2 = tf.get_variable(name='act_b2', shape=[36], initializer=tf.zeros_initializer)

            a_w3 = tf.get_variable(name='act_W3', shape=[5, 5, 36, 48],
                                   initializer=tf.contrib.layers.xavier_initializer())
            a_b3 = tf.get_variable(name='act_b3', shape=[48], initializer=tf.zeros_initializer)

            a_w4 = tf.get_variable(name='act_W4', shape=[3, 3, 48, 64],
                                   initializer=tf.contrib.layers.xavier_initializer())
            a_b4 = tf.get_variable(name='act_b4', shape=[64], initializer=tf.zeros_initializer)

            a_w5 = tf.get_variable(name='act_W5', shape=[3, 3, 64, 64],
                                   initializer=tf.contrib.layers.xavier_initializer())
            a_b5 = tf.get_variable(name='act_b5', shape=[64], initializer=tf.zeros_initializer)

            a_w_fc1 = tf.get_variable(name="act_W_fc1", shape=[1152, 1164],
                                      initializer=tf.contrib.layers.xavier_initializer())
            a_b_fc1 = tf.get_variable(name="act_b_fc1", shape=[1164],
                                      initializer=tf.contrib.layers.xavier_initializer())

            a_w_fc2 = tf.get_variable(name="act_W_fc2", shape=[1164, 100],
                                      initializer=tf.contrib.layers.xavier_initializer())
            a_b_fc2 = tf.get_variable(name="act_b_fc2", shape=[100],
                                      initializer=tf.contrib.layers.xavier_initializer())

            a_w_fc3 = tf.get_variable(name="act_W_fc3", shape=[100, 50],
                                      initializer=tf.contrib.layers.xavier_initializer())
            a_b_fc3 = tf.get_variable(name="act_b_fc3", shape=[50],
                                      initializer=tf.contrib.layers.xavier_initializer())

            a_w_fc4 = tf.get_variable(name="act_W_fc4", shape=[50, 10],
                                      initializer=tf.contrib.layers.xavier_initializer())
            a_b_fc4 = tf.get_variable(name="act_b_fc4", shape=[10],
                                      initializer=tf.contrib.layers.xavier_initializer())

            a_w_fc5 = tf.get_variable(name="act_W_fc5", shape=[10, env.action_space.n],
                                      initializer=tf.contrib.layers.xavier_initializer())
            a_b_fc5 = tf.get_variable(name="act_b_fc5", shape=[env.action_space.n],
                                      initializer=tf.contrib.layers.xavier_initializer())
        with tf.variable_scope("actor_conv_layers"):
            inp_batchnorm = tf.contrib.layers.batch_norm(state_inp, center=True, scale=True, is_training=True)
            conv1 = tf.nn.relu(tf.nn.conv2d(inp_batchnorm, a_w1, strides=[1, 2, 2, 1], padding='VALID') + a_b1)
            conv2 = tf.nn.relu(tf.nn.conv2d(conv1, a_w2, strides=[1, 2, 2, 1], padding='VALID') + a_b2)
            conv3 = tf.nn.relu(tf.nn.conv2d(conv2, a_w3, strides=[1, 2, 2, 1], padding='VALID') + a_b3)
            conv4 = tf.nn.relu(tf.nn.conv2d(conv3, a_w4, strides=[1, 1, 1, 1], padding='VALID') + a_b4)
            conv5 = tf.nn.relu(tf.nn.conv2d(conv4, a_w5, strides=[1, 1, 1, 1], padding='VALID') + a_b5)
        with tf.name_scope("actor_dense_Layers"):
            conv5_flattened = tf.contrib.layers.flatten(conv5)
            fc1 = tf.nn.relu(tf.matmul(conv5_flattened, a_w_fc1) + a_b_fc1)
            fc1 = tf.nn.dropout(fc1, keep_prob=conf.keep_prob)
            fc2 = tf.nn.relu(tf.matmul(fc1, a_w_fc2) + a_b_fc2)
            fc2 = tf.nn.dropout(fc2, keep_prob=conf.keep_prob)
            fc3 = tf.nn.relu(tf.matmul(fc2, a_w_fc3) + a_b_fc3)
            fc3 = tf.nn.dropout(fc3, keep_prob=conf.keep_prob)
            fc4 = tf.nn.relu(tf.matmul(fc3, a_w_fc4) + a_b_fc4)
            fc4 = tf.nn.dropout(fc4, keep_prob=conf.keep_prob)
        with tf.name_scope("actor_predictions"):
            out = tf.nn.softsign(tf.matmul(fc4, a_w_fc5) + a_b_fc5, name="actor_output")
            tf.summary.histogram('outputs', out)
        with tf.name_scope("actor_loss"):
            loss = tf.reduce_mean(tf.square(out - actor_action), name="sse_loss")
            tf.summary.scalar("actor_loss", loss)
        with tf.name_scope("actor_optimizer"):
            optim = tf.train.AdamOptimizer().minimize(loss)
    actor_nodes = {"state_inp": state_inp,
                   "action_inp": actor_action,
                   "out": out,
                   "loss": loss,
                   "optim": optim}
    return state_inp, actor_nodes


def supervised_train(nodes):
    print("inside supervised")
    max_epochs = conf.epochs
    batch_size = conf.batch_size
    with tf.Session() as sess:
        for epoch in range(1, max_epochs + 1):
            input_tensor = utils.get_batches()


def deep_q_train(nodes):
    """
    Main training loop to train the graph
    :param graph:
    :param nodes:
    :return:
    """
    keep_training = True
    with tf.Session() as sess:
        input_tensor = np.ones((conf.img_h, conf.img_w, conf.img_d))
        inp_t = np.stack((input_tensor, input_tensor, input_tensor, input_tensor), axis=0)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(conf.sum_dir + './train/', sess.graph)
        train_iter = 1
        epsilon = conf.initial_epsilon
        while keep_training:
            out_t = sess.run(nodes["out"], feed_dict={nodes["inp"]: inp_t})
            action_input = np.zeros([conf.OUTPUT_SIZE])
            # Perform random explore action or else grab maximum output
            if random.random() <= epsilon:
                act_indx = random.randrange(conf.OUTPUT_SIZE)
            else:
                act_indx = np.argmax(out_t)
            action_input[act_indx] = 1

            # TODO: write rest of reinforcement learning pipeline

            train_iter += 1
            if train_iter % conf.save_freq == 0:
                saver.save(sess, conf.save_dir + conf.save_name, global_step=train_iter)
            break
        train_writer.close()


def main():
    # graph, inp, max_action, optimal_action, out, action, loss, optimizer = create_graph()
    graph, nodes = create_graph()
    if conf.is_training:
        if conf.training_phase == 1:
            supervised_train(nodes)
        if conf.training_phase == 2:
            deep_q_train(nodes)


if __name__ == "__main__":
    main()






