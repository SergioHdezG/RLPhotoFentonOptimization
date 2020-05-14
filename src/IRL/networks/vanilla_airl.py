import random

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Conv1D, Flatten, MaxPooling2D, Dropout, Lambda, Input, Concatenate, Reshape, BatchNormalization, SimpleRNN, LSTM
from tensorflow.keras.optimizers import Adam
import numpy as np
import os.path as path

# Network for the Actor Critic
class Discriminator(object):
    def __init__(self, scope, state_size, n_actions, n_stack=False, img_input=False, expert_actions=False, learning_rate=1e-3, discrete=False):
        self.expert_actions = expert_actions

        self.state_size = state_size
        self.n_actions = n_actions

        self.img_input = img_input
        self.n_stack = n_stack
        self.stack = n_stack > 1
        self.learning_rate = learning_rate
        self.model = self._build_model()

        self.discrete_actions = discrete

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        if self.img_input:
            state_size = (*self.state_size[:2], self.state_size[-1] * self.n_stack)
        elif self.n_stack is not None and self.n_stack > 1:
            state_size = (self.n_stack, self.state_size)
        else:
            state_size = self.state_size

        s_input = Input(shape=state_size)
        a_input = Input(shape=self.n_actions)
        if self.stack:
            # flat_s = Dense(128, activation='tanh')(s_input)
            flat_s = Conv1D(32, kernel_size=3, strides=2, padding='same', activation='tanh')(s_input)
            flat_s = Flatten()(flat_s)
        else:
            flat_s = s_input
        concat = Concatenate(axis=1)([flat_s, a_input])
        dense = Dense(256, activation='tanh')(concat)
        # dense = Dropout(0.3)(dense)
        dense = Dense(256, activation='tanh')(dense)
        # dense = Dropout(0.4)(dense)
        dense = Dense(128, activation='tanh')(dense)
        output = Dense(1, activation='sigmoid')(dense)

        model = Model(inputs=[s_input, a_input], outputs=output)

        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    def get_reward(self, obs, action, asyncr=False):
        if asyncr:
            if self.discrete_actions:
                # One hot encoding
                action_matrix = np.array([np.zeros(self.n_actions) for a in action])
                for i in range(action.shape[0]):
                    action_matrix[i][action[i]] = 1
                action = action_matrix

            action = np.array(action)
            reward = self.model.predict([obs, action])
            reward = np.array([r[0] for r in reward])
            return reward
        else:
            if self.discrete_actions:
                # One hot encoding
                action_matrix = np.zeros(self.n_actions)
                action_matrix[action] = 1
                action = action_matrix

            action = np.array([action])
            # action = np.reshape(action, (1, -1))
            # obs = [np.concatenate((obs, action), axis=0)]
            # obs = np.array([np.concatenate((o, a), axis=0) for o, a in zip(obs, action)])
            # obs = np.reshape(np.concatenate((obs, action), axis=0), (1, -1))
            if self.stack:
                # obs = np.transpose(obs)
                # obs = np.reshape(obs, (1, self.state_size, self.n_stack))
                obs = np.array([obs])
                return self.model.predict([obs, action])[0]
            else:
                if len(obs.shape) > 1:
                    obs = obs[-1, :]
                # obs = np.reshape(obs, (1, -1))
                obs = np.array([obs])
                return self.model.predict([obs, action])[0]

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
                # expert_traj = [np.concatenate((x[0], x[1][0])) for x in expert_traj]
                # agent_traj = [np.concatenate((x[0], x[1][0])) for x in agent_traj]

                # # Take the same number of samples of each class
                # n_samples = min(len(expert_traj), len(agent_traj))
                # expert_traj = np.array(random.sample(expert_traj, n_samples))
                # agent_traj = np.array(random.sample(agent_traj, n_samples))
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

        # # Generating the training set
        # expert_label = np.ones((expert_traj.shape[0], 1))
        # agent_label = np.zeros((agent_traj.shape[0], 1))
        # x = np.concatenate([expert_traj, agent_traj], axis=0)
        # y = np.concatenate([expert_label, agent_label], axis=0)

        # Generating the training set
        expert_label = np.ones((expert_traj_s.shape[0], 1))
        agent_label = np.zeros((agent_traj_s.shape[0], 1))

        x_s = np.concatenate([expert_traj_s, agent_traj_s], axis=0)
        x_a = np.concatenate([expert_traj_a, agent_traj_a], axis=0)

        if self.discrete_actions:
            # One hot encoding
            one_hot_x_a = []
            for i in range(x_a.shape[0]):
                action_matrix = np.zeros(self.n_actions)
                action_matrix[x_a[i]] = 1
                one_hot_x_a.append(action_matrix)
            x_a = np.array(one_hot_x_a)

        y = np.concatenate([expert_label, agent_label], axis=0)

        self.model.fit([x_s, x_a], y, batch_size=128, epochs=10, shuffle=True, verbose=2, validation_split=0.2)

    # def load(self, dir, name):
    #     name = path.join(dir, name)
    #     loaded_model = tf.train.import_meta_graph(name)
    #     loaded_model.restore(self.sess, tf.train.latest_checkpoint(dir+"./"))
