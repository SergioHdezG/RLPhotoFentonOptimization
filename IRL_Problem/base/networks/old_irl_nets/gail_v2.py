"https://github.com/nav74neet/gail_gym/blob/03f6bb94c91e7cb7fee34c713d299d846ee28919/gail-ppo-tf-gym/network_models/discriminator.py"

import tensorflow as tf
import numpy as np
import random


class Discriminator:
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
        self.model = self._build_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.discrete_actions = discrete

    def _build_graph(self):
        with tf.variable_scope('discriminator'):
            self.scope = tf.get_variable_scope().name
            if self.stack:
                self.agent_traj_s = tf.placeholder(dtype=tf.float32, shape=(None, self.state_size, self.n_stack),
                                                   name="agent_traj_s")
                self.expert_traj_s = tf.placeholder(dtype=tf.float32, shape=(None, self.state_size, self.n_stack),
                                                    name="expert_traj_s")
            else:
                self.agent_traj_s = tf.placeholder(dtype=tf.float32, shape=(None, self.state_size), name="agent_traj_s")
                self.expert_traj_s = tf.placeholder(dtype=tf.float32, shape=(None, self.state_size),
                                                    name="expert_traj_s")

            self.agent_traj_a = tf.placeholder(dtype=tf.float32, shape=(None, self.n_actions), name="agent_traj_a")
            self.expert_traj_a = tf.placeholder(dtype=tf.float32, shape=(None, self.n_actions), name="expert_traj_a")

            self.expert_traj = tf.concat([self.agent_traj_s, self.expert_traj_a], axis=1)
            self.agent_traj = tf.concat([self.agent_traj_s, self.agent_traj_a], axis=1)

            # expert_a_one_hot = tf.one_hot(self.expert_a, depth=self.n_actions)
            # add noise for stabilise training
            # expert_a_one_hot += tf.random_normal(tf.shape(expert_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2
            # expert_s_a = tf.concat([self.expert_s, self.expert_a], axis=1)

            # self.agent_s = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.state_size))
            # self.agent_a = tf.placeholder(dtype=tf.int32, shape=[None])
            # agent_a_one_hot = tf.one_hot(self.agent_a, depth=self.n_actions)
            # add noise for stabilise training
            # agent_a_one_hot += tf.random_normal(tf.shape(agent_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2
            # agent_s_a = tf.concat([self.agent_s, agent_a_one_hot], axis=1)

            with tf.variable_scope('network') as network_scope:
                prob_1 = self.construct_network(input=self.expert_traj)
                network_scope.reuse_variables()  # share parameter
                prob_2 = self.construct_network(input=self.agent_traj)

            with tf.variable_scope('loss'):
                loss_expert = tf.reduce_mean(tf.log(tf.clip_by_value(prob_1, 0.01, 1)))
                loss_agent = tf.reduce_mean(tf.log(tf.clip_by_value(1 - prob_2, 0.01, 1)))
                loss = loss_expert + loss_agent
                self.loss = -loss
                tf.summary.scalar('discriminator', loss)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss)

            self.rewards = tf.log(tf.clip_by_value(prob_2, 1e-10, 1))  # log(P(expert|s,a)) larger is better for agent
            # self.rewards = -tf.log(1 - prob_2 + 1e-10)  # log(P(expert|s,a)) larger is better for agent

    def construct_network(self, input):
        # layer_1 = tf.layers.dense(inputs=input, units=256, activation=tf.nn.relu, name='layer1')
        layer_2 = tf.layers.dense(inputs=input, units=128, activation=tf.nn.relu, name='layer2')
        layer_3 = tf.layers.dense(inputs=layer_2, units=128, activation=tf.nn.relu, name='layer3')
        prob = tf.layers.dense(inputs=layer_3, units=1, activation=tf.sigmoid, name='prob')
        return prob

    # def train(self, expert_s, expert_a, agent_s, agent_a):
    def train(self, agent_traj, expert_traj):
        print("Training discriminator")

        # Formating network input
        if self.expert_actions:
            if self.img_input:
                pass
            elif self.stack:
                expert_traj_s = [x[0] for x in expert_traj]
                expert_traj_a = [x[1][0][0] for x in expert_traj]
                agent_traj_s = [x[0] for x in agent_traj]
                agent_traj_a = [x[1][0] for x in agent_traj]

                agent_traj = []
                expert_traj = []
                for i in range(len(agent_traj_s) - self.n_stack):
                    traj_stack = np.array([agent_traj_s[j] for j in range(i, i + self.n_stack)])
                    agent_traj.append(np.transpose(traj_stack))
                agent_traj_s = agent_traj

                for i in range(len(expert_traj_s) - self.n_stack):
                    traj_stack = np.array([expert_traj_s[j] for j in range(i, i + self.n_stack)])
                    expert_traj.append(np.transpose(traj_stack))
                expert_traj_s = expert_traj

            else:
                agent_traj_s = [x[0] for x in agent_traj]
                agent_traj_a = [x[1][0] for x in agent_traj]

                expert_traj_s = [x[0] for x in expert_traj]
                expert_traj_a = [x[1][0] for x in expert_traj]

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

        self.fit(expert_traj_s, expert_traj_a, agent_traj_s, agent_traj_a, batch_size=128, epochs=10, validation_split=0.2)
        # return self.sess.run(self.train_op, feed_dict={self.expert_traj: expert_traj,
        #                                                               self.agent_traj: agent_traj})

    def get_reward(self, obs, action, asyncr=False):
        if self.discrete_actions:
            # One hot encoding
            action_matrix = np.zeros(self.n_actions)
            action_matrix[action] = 1
            action = action_matrix

        action = np.array(action)
        if self.stack:
            obs = np.concatenate((np.reshape(obs, (-1,)), action))
            obs = np.reshape(obs, (1, -1))
            # obs = np.array([np.concatenate((np.reshape(o, (-1,)), a if action.shape[0] > 1 else [a])) for o, a in zip(obs, action)])
        else:
            obs = np.reshape(np.concatenate((obs, action), axis=0), (1, -1))

        return self.sess.run(self.rewards, feed_dict={self.agent_traj: obs})[0]

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

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

            val_loss = self.sess.run([self.loss], feed_dict={self.expert_traj_s: val_expert_traj_s,
                                                             self.expert_traj_a: val_expert_traj_a,
                                                             self.agent_traj_s: val_agent_traj_s,
                                                             self.agent_traj_a: val_agent_traj_a})[0]
            mean_loss = np.mean(mean_loss)
            print('epoch', epoch, " loss: ", mean_loss, " val_loss: ", val_loss)