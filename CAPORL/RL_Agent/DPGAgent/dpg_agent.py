from os import path

import tensorflow as tf
import numpy as np
from CAPORL.utils.parse_utils import *
from CAPORL.RL_Problem.rl_problem_super import *
from CAPORL.utils import net_building
from CAPORL.utils.networks import dpg_net

def create_agent():
    return 'DPG'


class Agent:
    """
    Deterministic Policy Gradient Agent
    """

    def __init__(self, n_actions, state_size, stack=False, img_input=False, learning_rate=0.01, gamma=0.95,
                 model_params=None, net_architecture=None):
        """
        Attributes:
            n_actions:          Int. Number of different actions.
            state_size:         Int or Tuple. State dimensions.
            learning_rate:      Learning rate for training.
            gamma:              Discount factor for target value.
            stack:              True if stacked inputs are used, False otherwise.
            img_input:          True if inputs are images, False otherwise.
            model_params:       Dictionary of params like learning rate, batch size, epsilon values, n step returns...
        """
        self.state_size = state_size
        self.n_actions = n_actions

        self.stack = stack
        self.img_input = img_input

        if model_params is not None:
            _, _, _, _, learning_rate, _ = \
                parse_model_params(model_params)

        self.lr = learning_rate
        self.gamma = gamma

        # initialize the memory for storing observations, actions and rewards
        self.memory = []
        self.done = False

        self._build_graph(net_architecture)

        self.epsilon = 0.  # Is not used here

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def remember(self, obs, action, reward, s_, done):
        # store actions as list of arrays
        action_one_hot = np.zeros(self.n_actions)
        action_one_hot[action] = 1
        self.done = done
        """
                Store a memory in a list of memories
                :param obs: Current Observation (State)
                :param action: Action selected
                :param reward: Reward
                :param next_obs: Next Observation (Next State)
                :param done: If the episode is finished
                :return:
                """
        self.memory.append([obs, action_one_hot, reward])

    def act(self, observation):
        """
        Select an action depending on the input type
        """
        if self.img_input:
            observation = np.squeeze(observation, axis=3)
            observation = observation.transpose(1, 2, 0)
            observation = np.array([observation])

        elif self.stack:
            observation = np.array([observation])
        else:
            observation = observation.reshape(-1, self.state_size)

        # TODO: aquí hay un problema de memoria
        prob_weights = self.sess.run(self.outputs_softmax, feed_dict={self.X: observation})
        action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())

        return action

    def act_test(self, observation):
        return self.act(observation)

    def _build_model(self, s, net_architecture):

        if net_architecture is None:  # Standart architecture
            net_architecture = dpg_net

        if self.img_input:
            model = net_building.build_conv_net(net_architecture, self.state_size)
            head = model(s)
            # conv1 = tf.keras.layers.Conv2D(32, input_shape=self.state_size, kernel_size=9, strides=(4, 4),
            #                                padding='same', activation='relu', name="conv_1")(s)
            # conv2 = tf.keras.layers.Conv2D(64, kernel_size=5, strides=(2, 2), padding='same', activation='relu',
            #                                name="conv_2")(conv1)
            # conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=(2, 2), padding='same', activation='relu',
            #                                name="conv_3")(conv2)
            # flat = tf.keras.layers.Flatten()(conv3)
            # head = tf.keras.layers.Dense(128, activation='relu')(flat)
        elif self.stack:
            model = net_building.build_stack_net(net_architecture, self.state_size)
            head = model(s)
            # flat = tf.keras.layers.Flatten(input_shape=self.state_size)(s)
            # head = tf.keras.layers.Dense(128, activation='relu')(flat)
        else:
            model = net_building.build_nn_net(net_architecture, self.state_size)
            head = model(s)
            # head = tf.keras.layers.Dense(128, input_dim=self.state_size, activation='relu')(s)

        # l_dense_2 = tf.keras.layers.Dense(256, activation='relu')(head)
        # out_actions = tf.keras.layers.Dense(self.n_actions, activation='linear')(l_dense_2)
        out_actions = tf.keras.layers.Dense(self.n_actions, activation='linear')(head)

        return out_actions

    def _build_graph(self, net_architecture):
        if self.img_input:  # and (self.stack ot not self.stack)
            # placeholders for input x, and output y
            self.X = tf.placeholder(tf.float32, shape=(None, *self.state_size), name="X")
            self.Y = tf.placeholder(tf.float32, shape=(None, self.n_actions), name="Y")
        elif self.stack:
            # placeholders for input x, and output y
            self.X = tf.placeholder(tf.float32, shape=(None, *self.state_size), name="X")
            self.Y = tf.placeholder(tf.float32, shape=(None, self.n_actions), name="Y")
        else:
            # placeholders for input x, and output y
            self.X = tf.placeholder(tf.float32, shape=(None, self.state_size), name="X")
            self.Y = tf.placeholder(tf.float32, shape=(None, self.n_actions), name="Y")

        # placeholder for reward
        self.discounted_episode_rewards_norm = tf.placeholder(tf.float32, [None, ], name="actions_value")

        logits = self._build_model(self.X, net_architecture)
        labels = self.Y
        self.outputs_softmax = tf.nn.softmax(logits, name='softmax')
        # self.outputs_softmax = tf.keras.activations.softmax(logits)

        # next we define our loss function as cross entropy loss
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        # reward guided loss
        loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_norm)

        # we use adam optimizer for minimizing the loss
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def discount_and_norm_rewards(self, episode_rewards):
        discounted_episode_rewards = np.zeros_like(episode_rewards)
        # discounted_episode_rewards = np.zeros_like(self.episode_rewards)
        cumulative = 0
        for t in reversed(range(episode_rewards.size)):
            cumulative = cumulative * self.gamma + episode_rewards[t]
            discounted_episode_rewards[t] = cumulative

        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards) + 1e-10  # para evitar valores cero
        return discounted_episode_rewards

    # now we actually learn i.e train our network
    def load_memories(self):
        """
        Load a batch of memories
        :return: Current Observation, Action, Reward, Next Observation, If episode is finished
        """
        memory = np.array(self.memory)
        obs, action, reward = memory[:, 0], memory[:, 1], memory[:, 2]
        obs = np.array([x.reshape(self.state_size) for x in obs])
        action = np.array([x.reshape(self.n_actions) for x in action])
        self.memory = []
        return obs, action, reward

    def replay(self):
        if self.done:
            obs, actions, reward = self.load_memories()
            # discount and normalize episodic reward
            discounted_episode_rewards_norm = self.discount_and_norm_rewards(reward)

            # train the nework
            dict = {self.X: obs,
                    self.Y: actions,
                    self.discounted_episode_rewards_norm: discounted_episode_rewards_norm,
                    }
            self.sess.run(self.train_op, feed_dict=dict)

            self.done = False
            return discounted_episode_rewards_norm

    def copy_model_to_target(self):
        pass

    def load(self, dir, name):
        name = path.join(dir, name)
        loaded_model = tf.train.import_meta_graph(name + '.meta')
        loaded_model.restore(self.sess, tf.train.latest_checkpoint(dir+"./"))

    def save(self, name, reward):
        self.saver.save(self.sess, name, global_step=reward)
        print("Saved model to disk")