import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Conv1D, Flatten, MaxPooling2D, Dropout, Lambda, Input, Concatenate, Reshape, BatchNormalization, SimpleRNN, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import numpy as np
import random

def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)

""" Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51"""
def logit_bernoulli_entropy(logits):
    ent = (1.-tf.nn.sigmoid(logits))*logits - logsigmoid(logits)
    return ent

class Discriminator(object):
    def __init__(self, scope, state_size, n_actions, n_stack=1, img_input=False, expert_actions=False,
                 learning_rate=1e-3, discrete=False):
        """
        :param env:
        Output of this Discriminator is reward for learning agent. Not the cost.
        Because discriminator predicts  P(expert|s,a) = 1 - P(agent|s,a).
        """
        self.expert_actions = expert_actions

        self.state_size = state_size
        self.n_actions = n_actions

        self.img_input = img_input
        self.n_stack = n_stack
        self.stack = self.n_stack > 1
        self.learning_rate = learning_rate
        self.entropy_beta = 0.001
        self._build_graph()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.discrete_actions = discrete

    def _build_graph(self):

        expert_net, agent_net = self._build_model()

        sig_expert_net = tf.keras.activations.sigmoid(expert_net)
        sig_agent_net = tf.keras.activations.sigmoid(agent_net)

        with tf.variable_scope('loss'):
            agent_expectation = tf.reduce_mean(tf.log(tf.clip_by_value(sig_agent_net, 1e-5, 1)))
            expert_expectation = tf.reduce_mean(tf.log(tf.clip_by_value(1 - sig_expert_net, 1e-5, 1)))

            # steer_mean, acc_mean, brk_mean = tf.split(self.agent_traj_a, [1, 1, 1], axis=1)
            # std = tf.math.reduce_std(self.agent_traj_a, axis=0, keepdims=True)+1e-5

            # entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))

            # normal_dist = tf.contrib.distributions.Normal(self.agent_traj_a, std)
            # entropy = tf.reduce_mean(normal_dist.entropy())

            mean = tf.reshape(self.agent_traj_a, [-1])
            _, _, count = tf.unique_with_counts(mean)
            prob = count / tf.reduce_sum(count)
            entropy = -tf.reduce_sum(prob * tf.log(prob))


            expectation = agent_expectation + expert_expectation #- self.entropy_beta*entropy
            self.loss = -expectation

        self.expectations = [agent_expectation, expert_expectation, entropy]
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

        self.reward = -sig_agent_net + 1
        # self.reward2 = -sig_expert_net + 1
        # self.reward = -tf.log(tf.clip_by_value(sig_agent_net, 1e-4, 1))  # log(P(expert|s,a)) larger is better for agent
        # self.reward = tf.log(1 - sig_agent_net + 1e-1)  # log(P(expert|s,a)) larger is better for agent

    def _build_model(self):
        with tf.variable_scope('discriminator'):
            self.scope = tf.get_variable_scope().name
            if self.stack:
                self.agent_traj_s = tf.placeholder(dtype=tf.float32, shape=(None, self.n_stack, self.state_size),
                                                   name="agent_traj_s")
                self.expert_traj_s = tf.placeholder(dtype=tf.float32, shape=(None, self.n_stack, self.state_size),
                                                    name="expert_traj_s")
            else:
                self.agent_traj_s = tf.placeholder(dtype=tf.float32, shape=(None, self.state_size), name="agent_traj_s")
                self.expert_traj_s = tf.placeholder(dtype=tf.float32, shape=(None, self.state_size),
                                                    name="expert_traj_s")

            self.agent_traj_a = tf.placeholder(dtype=tf.float32, shape=(None, self.n_actions), name="agent_traj_a")
            self.expert_traj_a = tf.placeholder(dtype=tf.float32, shape=(None, self.n_actions), name="expert_traj_a")

            if self.stack:
                # expert_traj_s = Conv1D(64, kernel_size=3, strides=2, padding='same', activation='tanh')(self.expert_traj_s)
                # expert_traj_s = Flatten()(self.expert_traj_s)
                expert_traj_s = LSTM(256, activation='tanh')(self.expert_traj_s)

                # agent_traj_s = Conv1D(64, kernel_size=3, strides=2, padding='same', activation='tanh')(self.agent_traj_s)
                # agent_traj_s = Flatten()(self.agent_traj_s)
                agent_traj_s = LSTM(256, activation='tanh')(self.agent_traj_s)
                self.expert_traj = tf.concat([expert_traj_s, self.expert_traj_a], axis=1)
                self.agent_traj = tf.concat([agent_traj_s, self.agent_traj_a], axis=1)
            else:
                self.expert_traj = tf.concat([self.expert_traj_s, self.expert_traj_a], axis=1)
                self.agent_traj = tf.concat([self.agent_traj_s, self.agent_traj_a], axis=1)

            with tf.variable_scope('network') as network_scope:
                discriminator = Sequential()
                # discriminator.add(Dense(2048, activation='tanh'))
                # discriminator.add(Dropout(0.4))
                # discriminator.add(Dense(256, activation='tanh'))
                # discriminator.add(Dropout(0.4))
                # discriminator.add(Dense(256, activation='tanh'))
                # discriminator.add(Dropout(0.3))
                discriminator.add(Dense(128, activation='tanh'))
                discriminator.add(Dense(1, activation='linear'))

                expert_net = discriminator(self.expert_traj)
                agent_net = discriminator(self.agent_traj)
        return expert_net, agent_net

    def get_reward(self, obs, action, asyncr=False):
        if asyncr:
            if self.discrete_actions:
                # One hot encoding
                action_matrix = np.array([np.zeros(self.n_actions) for a in action])
                for i in range(action.shape[0]):
                    action_matrix[i][action[i]] = 1
                action = action_matrix

            reward = self.sess.run(self.reward, feed_dict={self.agent_traj_s: obs,
                                                           self.agent_traj_a: action})
            reward = np.array([r[0] for r in reward])
            return reward
        else:
            if self.discrete_actions:
                # One hot encoding
                action_matrix = np.zeros(self.n_actions)
                action_matrix[action] = 1
                action = action_matrix

            action = np.array([action])
            if self.stack:
                # obs = np.transpose(obs)
                # obs = np.reshape(obs, (1, self.state_size, self.n_stack))
                obs = np.array([obs])
            else:
                if len(obs.shape) > 1:
                    obs = obs[-1, :]
                obs = np.array([obs])
                # If inputs are stacked but nor the discriminator, select the las one input from each stack


            reward = self.sess.run(self.reward, feed_dict={self.agent_traj_s: obs,
                                                           self.agent_traj_a: action})[0]
            # reward2 = self.sess.run(self.reward2, feed_dict={self.expert_traj_s: obs,
            #                                                  self.expert_traj_a: action})[0]
            # print('Reward_1: ', reward, 'reward_2: ', reward2)
        return reward

    # def train(self, expert_s, expert_a, agent_s, agent_a):
    def train(self, expert_traj, agent_traj):
        print("Training discriminator")

        # Formating network input
        if self.expert_actions:
            if self.img_input:
                pass
            elif self.stack:
                expert_traj_s = [np.array(x[0]) for x in expert_traj]
                expert_traj_a = [x[1] for x in expert_traj]
                agent_traj_s = [np.array(x[0]) for x in agent_traj]
                agent_traj_a = [x[1] for x in agent_traj]

            else:
                agent_traj_s = [x[0] for x in agent_traj]
                agent_traj_a = [x[1] for x in agent_traj]

                # If inputs are stacked but nor the discriminator, select the las one input from each stack
                if len(agent_traj_s[0].shape) > 1:
                    agent_traj_s = [x[-1, :] for x in agent_traj_s]

                expert_traj_s = [x[0] for x in expert_traj]
                expert_traj_a = [x[1] for x in expert_traj]



        # Take the same number of samples of each class
        n_samples = min(len(expert_traj), len(agent_traj))
        expert_traj_index = np.array(random.sample(range(len(expert_traj)), n_samples))
        agent_traj_index = np.array(random.sample(range(len(agent_traj)), n_samples))

        expert_traj_s = np.array(expert_traj_s)[expert_traj_index]
        expert_traj_a = np.array(expert_traj_a)[expert_traj_index]

        agent_traj_s = np.array(agent_traj_s)[agent_traj_index]
        agent_traj_a = np.array(agent_traj_a)[agent_traj_index]

        if self.discrete_actions:
            # One hot encoding
            one_hot_agent_a = []
            one_hot_expert_a = []
            for i in range(agent_traj_a.shape[0]):
                action_matrix_agent = np.zeros(self.n_actions)
                action_matrix_expert = np.zeros(self.n_actions)

                action_matrix_agent[agent_traj_a[i]] = 1
                action_matrix_expert[expert_traj_a[i]] = 1

                one_hot_agent_a.append(action_matrix_agent)
                one_hot_expert_a.append(action_matrix_expert)

            agent_traj_a = np.array(one_hot_agent_a)
            expert_traj_a = np.array(one_hot_expert_a)

        loss = self.fit(expert_traj_s, expert_traj_a, agent_traj_s, agent_traj_a, batch_size=128, epochs=2,
                 validation_split=0.15)  # batch_size=expert_traj_a.shape[0]
        return loss

    def fit(self, expert_traj_s, expert_traj_a, agent_traj_s, agent_traj_a, batch_size=128, epochs=10, validation_split=0.2):
        test_samples = np.int(validation_split * expert_traj_s.shape[0])
        train_samples = np.int(expert_traj_s.shape[0] - test_samples)

        val_expert_traj_s = expert_traj_s[:test_samples]
        val_expert_traj_a = expert_traj_a[:test_samples]
        val_agent_traj_s = agent_traj_s[:test_samples]
        val_agent_traj_a = agent_traj_a[:test_samples]

        expert_traj_s = expert_traj_s[test_samples:]
        expert_traj_a = expert_traj_a[test_samples:]
        agent_traj_s = agent_traj_s[test_samples:]
        agent_traj_a = agent_traj_a[test_samples:]

        val_loss = 100
        print("train samples: ", train_samples*2, " val_samples: ", test_samples*2)
        for epoch in range(epochs):
            mean_loss = []
            for batch in range(train_samples//batch_size + 1):
                i = batch * batch_size
                j = (batch+1) * batch_size

                if j >= train_samples:
                    j = train_samples

                expert_batch_s = expert_traj_s[i:j]
                expert_batch_a = expert_traj_a[i:j]
                agent_batch_s = agent_traj_s[i:j]
                agent_batch_a = agent_traj_a[i:j]
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.expert_traj_s: expert_batch_s,
                                                                               self.expert_traj_a: expert_batch_a,
                                                                               self.agent_traj_s: agent_batch_s,
                                                                               self.agent_traj_a: agent_batch_a
                                                                               })
                mean_loss.append(loss)

            val_loss, expectations = self.sess.run([self.loss, self.expectations], feed_dict={self.expert_traj_s: val_expert_traj_s,
                                                             self.expert_traj_a: val_expert_traj_a,
                                                             self.agent_traj_s: val_agent_traj_s,
                                                             self.agent_traj_a: val_agent_traj_a})
            mean_loss = np.mean(mean_loss)
            print('epoch', epoch, "\tloss: ", mean_loss, "\tval_loss: ", val_loss, '\tagent_expectation: ', expectations[0],
                  '\texpert_expectations: ', expectations[1], '\tentropy: ', expectations[2])
        return val_loss