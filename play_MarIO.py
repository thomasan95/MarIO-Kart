from utilities import resize_img, XboxController
from train_MarIO import create_graph
import config
import numpy as np
from termcolor import cprint
import gym
import gym_mupen64plus
import tensorflow as tf
conf = config.Config()


class Actor(object):
    def __init__(self, sess):
        with tf.variable_scope("Actor_Graph"):
            self.state_inp, self.model = create_graph(keep_prob=1)
        self.sess = sess
        self.saver = tf.train.Saver()
        self.saver.restore(sess, conf.save_dir + conf.save_name)

        self.real_controller = XboxController()

    def get_action(self, obs):
        manual_override = self.real_controller.LeftBumper == 1
        if not manual_override:
            # Look
            vec = resize_img(obs)
            vec = np.expand_dims(vec, axis=0)  # expand dimensions for predict, it wants (1,66,200,3) not (66, 200, 3)
            # Think
            out = self.sess.run(self.model["out"], feed_dict={self.model["state_inp"]: vec})
            joystick = out[0]
        else:
            joystick = self.real_controller.read()
            joystick[1] *= -1  # flip y (this is in the config when it runs normally)
        output = [
            int(joystick[0] * 80),
            int(joystick[1] * 80),
            int(round(joystick[2])),
            int(round(joystick[3])),
            int(round(joystick[4])),
        ]
        if manual_override:
            cprint("Manual: " + str(output), 'yellow')
        else:
            cprint("AI: " + str(output), 'green')

        return output


if __name__ == "__main__":
    env = gym.make('Mario-Kart-Royal-Raceway-v0')
    state = env.reset()
    env.render()
    print('env ready!')
    with tf.Session() as s:
        actor = Actor(s)
        print('actor ready!')
        print('beginning episode loop')
        total_reward = 0
        end_episode = False
        while not end_episode:
            action = actor.get_action(state)
            observation, reward, end_episode, info = env.step(action)
            env.render()
            total_reward += reward
        print('end episode... total reward: ' + str(total_reward))
        observation = env.reset()
        print('env ready!')
        input('press <ENTER> to quit')
        env.close()

