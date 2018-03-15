import tensorflow as tf
import config
import numpy as np
import argparse
import random
import utilities as utils
import gym
import gym_mupen64plus
from collections import deque
from sklearn.model_selection import train_test_split
import pickle as pkl
import os
print("Using TensorFlow version: " + str(tf.__version__))
print("This code was developed in version: 1.6.0")
# Load Configs
conf = config.Config()
parser = argparse.ArgumentParser(description="specify task of the network")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-s", "--supervised", action='store_true', help="supervised training")
group.add_argument("-dqn", "--reinforcement", action='store_true', help="reinforcement learning")
parser.add_argument("-r", "--resume", action="store_true", help="resume training. Specify file path in config.py")
args = parser.parse_args()


def create_graph(keep_prob=conf.keep_prob):
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
    # Fill in shape later since we'll downsample and resize
    with tf.variable_scope("actor"):
        # actor = tf.Graph()
        # with actor.as_graph_def():
        with tf.variable_scope("actor_input"):
            state_inp = tf.placeholder(tf.float32, shape=conf.inp_shape, name="actor_state_input")
            reinforcement_inp = tf.placeholder(tf.float32, shape=conf.r_inp_shape, name="reinforcement_inp")
            supervised_act = tf.placeholder(tf.float32, shape=[None, conf.OUTPUT_SIZE], name="supervised_action")
            actor_action = tf.placeholder(tf.float32, shape=[None, conf.OUTPUT_SIZE], name="actor_action_ph")
            yj = tf.placeholder(tf.float32, shape=[None], name="yj")
        with tf.variable_scope("actor_kernels_weights"):
            a_w1 = tf.get_variable(name="act_W1", shape=[5, 5, 3, 24],
                                   initializer=tf.contrib.layers.xavier_initializer())
            a_b1 = tf.get_variable(name="act_b1", shape=[24], initializer=tf.zeros_initializer)

            r_w1 = tf.get_variable(name="r_W1", shape=[5, 5, 12, 24],
                                   initializer=tf.contrib.layers.xavier_initializer())
            r_b1 = tf.get_variable(name="r_b1", shape=[24], initializer=tf.zeros_initializer)

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
            a_b_fc1 = tf.get_variable(name="act_b_fc1", shape=[1164], initializer=tf.zeros_initializer)

            a_w_fc2 = tf.get_variable(name="act_W_fc2", shape=[1164, 100],
                                      initializer=tf.contrib.layers.xavier_initializer())
            a_b_fc2 = tf.get_variable(name="act_b_fc2", shape=[100], initializer=tf.zeros_initializer)

            a_w_fc3 = tf.get_variable(name="act_W_fc3", shape=[100, 50],
                                      initializer=tf.contrib.layers.xavier_initializer())
            a_b_fc3 = tf.get_variable(name="act_b_fc3", shape=[50], initializer=tf.zeros_initializer)

            a_w_fc4 = tf.get_variable(name="act_W_fc4", shape=[50, 10],
                                      initializer=tf.contrib.layers.xavier_initializer())
            a_b_fc4 = tf.get_variable(name="act_b_fc4", shape=[10], initializer=tf.zeros_initializer)

            a_w_fc5 = tf.get_variable(name="act_W_fc5", shape=[10, conf.OUTPUT_SIZE],
                                      initializer=tf.contrib.layers.xavier_initializer())
            a_b_fc5 = tf.get_variable(name="act_b_fc5", shape=[conf.OUTPUT_SIZE], initializer=tf.zeros_initializer)
        with tf.variable_scope("actor_conv_layers"):
            if args.supervised:
                inp_batchnorm = tf.contrib.layers.batch_norm(state_inp, center=True, scale=True, is_training=True)
                conv1 = tf.nn.relu(tf.nn.conv2d(state_inp, a_w1, strides=[1, 2, 2, 1], padding='VALID') + a_b1)
            else:
                inp_batchnorm = tf.contrib.layers.batch_norm(reinforcement_inp, center=True, scale=True, is_training=True)
                conv1 = tf.nn.relu(tf.nn.conv2d(reinforcement_inp, r_w1, strides=[1, 2, 2, 1], padding='VALID') + r_b1)
            conv2 = tf.nn.relu(tf.nn.conv2d(conv1, a_w2, strides=[1, 2, 2, 1], padding='VALID') + a_b2)
            conv3 = tf.nn.relu(tf.nn.conv2d(conv2, a_w3, strides=[1, 2, 2, 1], padding='VALID') + a_b3)
            conv4 = tf.nn.relu(tf.nn.conv2d(conv3, a_w4, strides=[1, 1, 1, 1], padding='VALID') + a_b4)
            conv5 = tf.nn.relu(tf.nn.conv2d(conv4, a_w5, strides=[1, 1, 1, 1], padding='VALID') + a_b5)
        with tf.name_scope("actor_dense_Layers"):
            conv5_flattened = tf.contrib.layers.flatten(conv5)
            fc1 = tf.nn.relu(tf.matmul(conv5_flattened, a_w_fc1) + a_b_fc1)
            fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)
            fc2 = tf.nn.relu(tf.matmul(fc1, a_w_fc2) + a_b_fc2)
            fc2 = tf.nn.dropout(fc2, keep_prob=keep_prob)
            fc3 = tf.nn.relu(tf.matmul(fc2, a_w_fc3) + a_b_fc3)
            fc3 = tf.nn.dropout(fc3, keep_prob=keep_prob)
            fc4 = tf.nn.relu(tf.matmul(fc3, a_w_fc4) + a_b_fc4)
            fc4 = tf.nn.dropout(fc4, keep_prob=keep_prob)
        with tf.name_scope("actor_predictions"):
            out = tf.nn.softsign(tf.matmul(fc4, a_w_fc5) + a_b_fc5, name="actor_output")
            supervised_loss = tf.sqrt(tf.reduce_sum(tf.square(out - supervised_act), axis=-1))
            action = tf.reduce_sum(tf.multiply(out, actor_action))
            tf.summary.histogram('outputs', out)
            tf.summary.scalar('action', action)
            tf.summary.scalar('supervised_loss', supervised_loss)
        with tf.name_scope("actor_loss"):
            loss = tf.reduce_mean(tf.square(action - yj), name="sse_loss")
            tf.summary.scalar("actor_loss", loss)
        with tf.name_scope("actor_optimizer"):
            optim_supervised = tf.train.AdamOptimizer().minimize(supervised_loss)
            optim_reinforcement = tf.train.AdamOptimizer().minimize(loss)
    actor_nodes = {"state_inp": state_inp,
                   "action_inp": actor_action,
                   "r_inp": reinforcement_inp,
                   "yj": yj,
                   "out": out,
                   "action": action,
                   "s_action": supervised_act,
                   "loss": loss,
                   "s_loss": supervised_loss,
                   "optim_r": optim_reinforcement,
                   "optim_s": optim_supervised}
    return state_inp, actor_nodes


def supervised_train(nodes):
    """
    Performs supervised training on the agent
    :param nodes: graph nodes
    :type nodes: dict
    :return: losses during the training cycle
    :rtype: list
    """
    print("\nSupervised Training\n")
    losses = {'train': [],
              'valid': []}
    x_list, y_list = [], []
    for npy in os.listdir('data'):
        ext = os.path.splitext(npy)[-1].lower()
        if not ext == '.npy':
            continue
        if npy[0] == 'X':
            x_list.append(npy)
        elif npy[0] == 'y':
            y_list.append(npy)
    x_list.sort()
    y_list.sort()
    with tf.Session() as sess:
        saver = tf.train.Saver()
        if args.resume:
            saver.restore(sess, conf.save_dir + conf.save_name)
        else:
            sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(conf.sum_dir + './train/', sess.graph)
        train_iter = 0
        for epoch in range(1, conf.epochs + 1):
            mean_loss = 0
            print("\nEpoch %d\n" % epoch)
            train_loss, val_loss = 0, 0
            indexes = np.arange(len(x_list))
            mean_loss = 0
            if conf.shuffle:
                np.random.shuffle(indexes)
            for num, file_i in enumerate(indexes):
                batch_size = conf.batch_size
                x_d = x_list[file_i]
                y_d = y_list[file_i]
                if not (x_d[2:] == y_d[2:]):
                    print("File not the same. They are: " + x_d[2:] + " and " + y_d[2:])
                    continue
                print("Loading " + str(num) + "th Race: " + x_d[2:])
                x_data, y_data = np.load(conf.data_dir + x_d), np.load(conf.data_dir + y_d)
                x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=conf.val_split)
                if len(x_train) < batch_size:
                    batch_size = len(x_train)
                    num_batches = 1
                    val_size = 1
                else:
                    num_batches = len(x_train) // batch_size
                    val_size = len(x_val) // batch_size
                for batch_i, (x_input, y_input) in enumerate(utils.get_batches(x_train, y_train, batch_size)):
                    loss, out, _ = sess.run([nodes["s_loss"], nodes["out"], nodes["optim_s"]],
                                            feed_dict={nodes["state_inp"]: x_input,
                                                       nodes["s_action"]: y_input})
                    print("[%.4f, %.4f, %.4f, %.4f, %.4f]" % (out[0,0], out[0,1], out[0,2], out[0,3], out[0,4]))
                    mean_loss += np.mean(loss)
                    train_loss += mean_loss
                    train_iter += 1
                    if train_iter % 50 == 0:
                        samp_out, samp_true = out[0], y_input[0]
                        diff = np.absolute(samp_out - samp_true)
                        print("Difference between out and true: ")
                        print("[%.4f, %.4f, %.4f, %.4f, %.4f]" % (float(diff[0]), float(diff[1]), float(diff[2]),
                                                                  float(diff[3]), float(diff[4])))
                        print("Done with %d iterations of %d training samples:\tCurr Loss: %f" %
                              (train_iter, batch_i*batch_size + batch_size, mean_loss/50))
                        mean_loss = 0
                    if train_iter % conf.save_freq == 0:
                        saver.save(sess, conf.save_dir + conf.save_name)
                if len(x_val) < batch_size:
                    batch_size = len(x_val)
                for val_i, (val_x_inp, val_y_inp) in enumerate(utils.get_batches(x_val, y_val, batch_size)):
                    loss = sess.run(nodes["s_loss"], feed_dict={nodes["state_inp"]: val_x_inp,
                                                                nodes["s_action"]: val_y_inp})
                    mean_loss = np.mean(loss)
                    val_loss += mean_loss
                # Append losses to generate plots in the future
                losses["train"].append(train_loss/num_batches)
                losses["valid"].append(val_loss/val_size)
                with open(conf.pickle_dir + 'losses.p', 'wb') as f:
                    pkl.dump(losses, f)
        # Close train writer
        train_writer.close()
        return losses


def deep_q_train(nodes):
    """
    Main training loop to train the graph
    :param nodes: Nodes for the graph so that the Tensorflow network can run
    :type nodes: dict{str: tf.Tensors}
    :return:
    """
    print("\nReinforcement Learning\n")
    env = gym.make('Mario-Kart-Royal-Raceway-v0')
    with tf.Session() as sess:
        saver = tf.train.Saver()
        # Initialize all variables such as Q inside network
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(conf.sum_dir + './train/', sess.graph)
        # Initialize memory to some capacity
        memory = deque(maxlen=conf.replay_memory)
        epsilon = conf.initial_epsilon

        for episode in range(1, conf.max_episodes):
            still_in_episode = True
            # Will replace with samples from the initial game screen
            # Want to send in 4 screens at a time to process, so stack along depth of image
            input_tensor = env.reset()
            input_tensor = utils.resize_img(input_tensor)
            inp = np.dstack((input_tensor, input_tensor, input_tensor, input_tensor))
            time_step = 0
            while still_in_episode:
                # Grab actions from first state
                action_input = np.zeros([conf.OUTPUT_SIZE])
                state = np.expand_dims(inp, axis=0)
                out_t = sess.run(nodes["out"], feed_dict={nodes["state_inp"]: state})[0]
                # Perform random explore action or else grab maximum output
                if random.random() <= epsilon:
                    act_indx = random.randrange(conf.OUTPUT_SIZE)
                else:
                    act_indx = np.argmax(out_t)
                action_input[act_indx] = 1
                # Randomness factor
                if epsilon > conf.final_epsilon:
                    epsilon *= conf.epsilon_decay
                # Observe next reward from action
                observation, reward, end_episode, info = env.step(action_input)
                # Finish rest of the pipeline for this time step, but proceed to the next episode after
                obs = utils.resize_img(observation)
                if end_episode:
                    still_in_episode = False
                env.render()
                obs = np.expand_dims(obs, axis=0)
                new_state = np.zeros(state.shape)
                new_state[:, :, :, :3] = obs
                new_state[:, :, :, 3:] = state[:, :, :, :9]
                # Add to memory
                memory.append((state, action_input, reward, new_state))

                if time_step > conf.start_memory_sample:
                    batch = random.sample(memory, conf.batch_size)
                    mem_state = [mem[0] for mem in batch]
                    mem_action = [mem[1] for mem in batch]
                    mem_reward = [mem[2] for mem in batch]
                    mem_next_state = [mem[3] for mem in batch]

                    yj = []
                    mem_out = sess.run(nodes["out"], feed_dict={nodes["state_inp"]: mem_next_state})
                    for i in range(0, len(batch)):
                        yj.append(mem_reward[i] + conf.learning_rate*np.max(mem_out[i]))

                    # Perform gradient descent on the loss function with respect to the yj and predicted output
                    _ = sess.run(nodes["optim_r"], feed_dict={nodes["yj"]: yj,
                                                              nodes["action_inp"]: mem_action,
                                                              nodes["state_inp"]: mem_state})
                state = new_state
                time_step += 1
                if time_step % conf.save_freq == 0:
                    saver.save(sess, conf.save_dir + conf.save_name, global_step=time_step)
                if time_step % 100 == 0:
                    print("Episode: %d, Time Step: %d, Reward: %d" % (episode, time_step, reward))
            train_writer.close()


def policy_gradient_train(nodes):
    env = gym.make('Mario-Kart-Royal-Raceway-v0')
    with tf.Session() as sess:
        saver = tf.train.Saver()
        if conf.resume_training:
            saver.restore(sess, conf.save_dir + conf.save_name)
        else:
            sess.run(tf.global_variables_initializer())
        state = env.reset()
        states, actions, rewards = [], [], []
        for episode in range(conf.max_episodes):
            reward = 0
            keep_training = True
            previous_state = np.zeros((conf.img_h, conf.img_w, conf.img_d))
            while keep_training:
                state = utils.resize_img(state)
                # Feed in the difference in states
                state_inp = state - previous_state
                previous_state = state
                out = sess.run(nodes["out"], feed_dict={nodes["state_inp"]: state_inp})
                states.append(state_inp)
                actions.append(out)
                state, r, end_episode, _ = env.step(out)
                reward += r
                rewards.append(reward)
                if end_episode:
                    keep_training = False


def main():
    # graph, inp, max_action, optimal_action, out, action, loss, optimizer = create_graph()
    with tf.variable_scope("Actor_Graph"):
        graph, nodes = create_graph()
    if conf.is_training:
        if args.supervised:
            losses = supervised_train(nodes)
        elif args.reinforcement:
            deep_q_train(nodes)


if __name__ == "__main__":
    main()






