from CNN import Net
import utilities as utils
import config

#import gym
#import gym_mupen64plus
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import random
from collections import deque
from sklearn.model_selection import train_test_split
import pickle as pkl
import os


   


# Load Configs
conf = config.Config()
parser = argparse.ArgumentParser(description="specify task of the network")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-s", "--supervised", action='store_true', help="supervised training")
group.add_argument("-dqn", "--reinforcement", action='store_true', help="reinforcement learning")
parser.add_argument("-r", "--resume", action="store_true", help="resume training. Specify file path in config.py")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Specify learning rate of the network")
args = parser.parse_args()

gpu = torch.cuda.is_available()

def supervised_train(model, criterion, optimizer):
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

    # Get folders for training data
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

    train_iter = 0
    for epoch in range(1, conf.epochs + 1):
        print("\nEpoch %d\n" % epoch)
        train_loss, val_loss = 0, 0
        indexes = np.arange(len(x_list))
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
                
                x_input = np.swapaxes(x_input,1,3)
                print np.shape(x_input)
                x_input,y_input = torch.from_numpy(x_input).float(),torch.from_numpy(y_input).float()
                x_input,y_input = Variable(x_input),Variable(y_input)
                output = model(x_input)
                loss = criterion(output,y_input)
                train_loss += loss
                train_iter += 1
                if train_iter % 10 == 0:
                    print("Done with %d iterations of training:\tCurr Loss: %f" % (train_iter, loss))

                if train_iter % conf.save_freq == 0:
                    if not os.path.isdir(config.save_dir):
                        os.mkdir(config.save_dir)
                    utils.save_checkpoint({'epoch': epoch, 
                                           'state_dict': model.state_dict(),
                                           'optimizer': optimizer.state_dict()},
                                            config.checkpoint)

            # Validation check
            if len(x_val) < batch_size:
                batch_size = len(x_val)

            for val_i, (val_x_inp, val_y_inp) in enumerate(utils.get_batches(x_val, y_val, batch_size)):
                val_x_inp = np.swapaxes(val_x_inp,1,3)
                print np.shape(val_x_inp)
                val_x_inp,val_y_inp = torch.from_numpy(val_x_inp).float(),torch.from_numpy(val_y_inp).float()
                val_x_inp,val_y_inp = Variable(val_x_inp),Variable(val_y_inp)
                output = model(val_x_inp)
                loss = criterion(output,val_y_inp)
                val_loss += loss

            # Append losses to generate plots in the future
            losses["train"].append(train_loss/num_batches)
            losses["valid"].append(val_loss/val_size)
            with open(conf.pickle_dir + 'losses.p', 'wb') as f:
                pkl.dump(losses, f)

    return losses


def deep_q_train(model):
    """
    Main training loop to train the graph
    :param nodes: Nodes for the graph so that the Tensorflow network can run
    :type nodes: dict{str: tf.Tensors}
    :return:
    """
    print("\nReinforcement Learning\n")
    env = gym.make('Mario-Kart-Royal-Raceway-v0')

    # Initialize memory to some capacity
    memory = deque(maxlen=conf.replay_memory)
    epsilon = conf.initial_epsilon

    for episode in range(1, conf.max_episodes):
        end_episode = False
        # Will replace with samples from the initial game screen
        # Want to send in 4 screens at a time to process, so stack along depth of image
        input_tensor = env.reset()
        input_tensor = utils.resize_img(input_tensor)
        inp = np.dstack((input_tensor, input_tensor, input_tensor, input_tensor))
        time_step = 0
        while not end_episode:
            # Grab actions from first state
            action_input = np.zeros([conf.OUTPUT_SIZE])
            state = np.expand_dims(inp, axis=0)
            output = model(state)
            # Perform random explore action or else grab maximum output
            if random.random() <= epsilon:
                act_indx = random.randrange(conf.OUTPUT_SIZE)
            else:
                act_indx = output.data.cpu().numpy()
            action_input[act_indx] = 1

            # Randomness factor
            if epsilon > conf.final_epsilon:
                epsilon *= conf.epsilon_decay

            # Observe next reward from action
            observation, reward, end_episode, info = env.step(action_input)
            # Finish rest of the pipeline for this time step, but proceed to the next episode after
            obs = utils.resize_img(observation)
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
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if conf.is_training:
        if args.supervised:
            model = Net()
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            criterion = torch.nn.MSELoss() 
            losses = supervised_train(model, criterion, optimizer)
        elif args.reinforcement:
            model, optimizer, _, losses = utils.resume_checkpoint(model,optimizer,gpu,config.checkpoint)
            deep_q_train(model)


if __name__ == "__main__":
    main()
