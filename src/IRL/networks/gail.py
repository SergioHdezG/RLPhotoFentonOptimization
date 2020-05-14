'''
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
'''
import tensorflow as tf
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
                 learning_rate=1e-4, discrete=False):
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

    def _build_graph(self, entcoeff=0.001, scope="adversary"):
        self.scope = scope
        self.observation_shape = self.state_size
        self.build_ph()


        # Build grpah
        generator_logits = self.build_net(self.agent_traj_s, self.agent_traj_a, reuse=False)
        expert_logits = self.build_net(self.expert_traj_s, self.expert_traj_a, reuse=True)
        # Build accuracy
        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
        expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))
        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits, labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)
        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff*entropy
        # Loss + Accuracy terms
        self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
        self.total_loss = generator_loss + expert_loss + entropy_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.total_loss)

        # Build Reward for policy
        self.reward_op = -tf.log(1-tf.nn.sigmoid(generator_logits)+1e-8)
        # self.reward_op = tf.log(tf.nn.sigmoid(generator_logits) + 1e-8)
        # self.reward_op = tf.nn.sigmoid(generator_logits)

        # var_list = self.get_trainable_variables()
        # self.lossandgrad = U.function([self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph],
        #                               self.losses + [U.flatgrad(self.total_loss, var_list)])

    def build_ph(self):
        if self.stack:
            self.agent_traj_s = tf.placeholder(dtype=tf.float32, shape=(None, self.state_size, self.n_stack), name="agent_traj_s")
            self.expert_traj_s = tf.placeholder(dtype=tf.float32, shape=(None, self.state_size, self.n_stack), name="expert_traj_s")
        else:
            self.agent_traj_s = tf.placeholder(dtype=tf.float32, shape=(None, self.state_size), name="agent_traj_s")
            self.expert_traj_s = tf.placeholder(dtype=tf.float32, shape=(None, self.state_size), name="expert_traj_s")

        self.agent_traj_a = tf.placeholder(dtype=tf.float32, shape=(None, self.n_actions), name="agent_traj_a")
        self.expert_traj_a = tf.placeholder(dtype=tf.float32, shape=(None, self.n_actions), name="expert_traj_a")

    def build_net(self, obs_ph, action_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("obfilter"):
                s_mean = tf.to_float(tf.reduce_mean(obs_ph, 1, keepdims=True))
                s_std = tf.math.reduce_std(obs_ph, 1, keepdims=True)
                state = (obs_ph - s_mean) / (s_std + 1e-10)
                obs_ph = tf.concat([state, action_ph], axis=1)  # concatenate the two input -> form a transition

            p_h1 = tf.keras.layers.Dense(128, activation='relu')(obs_ph)
            p_h1 = tf.keras.layers.Dropout(0.3)(p_h1)
            p_h2 = tf.keras.layers.Dense(128, activation='relu')(p_h1)
            logits = tf.keras.layers.Dense(1, activation='linear')(p_h2)
        return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs, action, asyncr=False):
        if self.discrete_actions:
            # One hot encoding
            action_matrix = np.zeros(self.n_actions)
            action_matrix[action] = 1
            action = action_matrix

        if asyncr:
            action = np.array(action)
            obs = np.array([np.concatenate((o, a if action.shape[1] > 1 else [a]), axis=0) for o, a in zip(obs, action)])
            # obs = np.reshape(np.concatenate((obs, action), axi1), (action.shape[0], -1))
            reward = self.model.predict(obs)
            reward = np.array([r[0] for r in reward])
            return reward
        else:
            action = np.reshape(action, (1, -1))
            if self.stack:
                obs = np.transpose(obs)
                obs = np.reshape(obs, (1, self.state_size, self.n_stack))
            else:
                obs = np.reshape(obs, (1, -1))
        return self.sess.run(self.reward_op, feed_dict={self.agent_traj_s: obs,
                                                        self.agent_traj_a: action})[0]

    # def train(self, expert_s, expert_a, agent_s, agent_a):
    def train(self, expert_traj, agent_traj):
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
                    traj_stack = np.array([agent_traj_s[j] for j in range(i, i+self.n_stack)])
                    agent_traj.append(np.transpose(traj_stack))
                agent_traj_s = agent_traj

                for i in range(len(expert_traj_s) - self.n_stack):
                    traj_stack = np.array([expert_traj_s[j] for j in range(i, i+self.n_stack)])
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

        self.fit(expert_traj_s, expert_traj_a, agent_traj_s, agent_traj_a, batch_size=128, epochs=2, validation_split=0.2)

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
                _, loss = self.sess.run([self.train_op, self.total_loss], feed_dict={self.expert_traj_s: expert_batch_s,
                                                                                     self.expert_traj_a: expert_batch_a,
                                                                                     self.agent_traj_s: agent_batch_s,
                                                                                     self.agent_traj_a: agent_batch_a,
                                                                                     })

                if np.isnan(loss):
                    print("what???")
                mean_loss.append(loss)

            train_loss = self.sess.run(self.losses, feed_dict={self.expert_traj_s: expert_batch_s,
                                                               self.expert_traj_a: expert_batch_a,
                                                               self.agent_traj_s: agent_batch_s,
                                                               self.agent_traj_a: agent_batch_a,
                                                               })
            val_loss = self.sess.run(self.losses, feed_dict={self.expert_traj_s: val_expert_traj_s,
                                                             self.expert_traj_a: val_expert_traj_a,
                                                             self.agent_traj_s: val_agent_traj_s,
                                                             self.agent_traj_a: val_agent_traj_a,
                                                             })
            # mean_loss = np.mean(mean_loss)
            print('train:\t\t', self.loss_name[0], train_loss[0], "\t", self.loss_name[1], train_loss[1], "\t",
                  self.loss_name[2], train_loss[2], "\t", self.loss_name[3], train_loss[3], "\t", self.loss_name[4],
                  train_loss[4], "\t", self.loss_name[5], "\t", train_loss[5])

            print('validation:\t', self.loss_name[0], val_loss[0], "\t", self.loss_name[1],  val_loss[1], "\t", self.loss_name[2],  val_loss[2],
                  "\t", self.loss_name[3],  val_loss[3], "\t", self.loss_name[4],  val_loss[4], "\t",
                  self.loss_name[5],  "\t", val_loss[5])