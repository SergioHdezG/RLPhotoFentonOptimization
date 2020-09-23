import random
from IRL_Problem.base.networks.discriminator_base import DiscriminatorBase
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Conv1D, Flatten, MaxPooling2D, Dropout, Lambda, Input, Concatenate, Reshape, BatchNormalization, SimpleRNN, LSTM
from tensorflow.keras.optimizers import Adam
import numpy as np
import os.path as path

# Network for the Actor Critic
class Discriminator(DiscriminatorBase):
    def __init__(self, scope, state_size, n_actions, n_stack=False, img_input=False, expert_actions=False,
                 learning_rate=1e-3, discrete=False):
        super().__init__(scope=scope, state_size=state_size, n_actions=n_actions, n_stack=n_stack, img_input=img_input,
                         expert_actions=expert_actions, learning_rate=learning_rate, discrete=discrete)
        self.model = self._build_model()



    def _build_net(self, state_size):
        s_input = Input(shape=state_size)
        a_input = Input(shape=self.n_actions)
        if self.stack:
            # flat_s = Dense(128, activation='tanh')(s_input)
            # flat_s = Conv1D(32, kernel_size=3, strides=2, padding='same', activation='tanh')(s_input)
            flat_s = LSTM(32, activation='tanh')(s_input)
            # flat_s = Flatten()(s_input)
        else:
            flat_s = s_input
        concat = Concatenate(axis=1)([flat_s, a_input])
        # dense = Dropout(0.4)(concat)
        dense = Dense(64, activation='relu')(concat)
        # dense = Dropout(0.4)(dense)
        # dense = Dense(256, activation='tanh')(dense)
        dense = Dense(64, activation='relu')(dense)
        # dense = concat
        output = Dense(1, activation='sigmoid')(dense)
        model = Model(inputs=[s_input, a_input], outputs=output)
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    def predict(self, obs, action):
        return self.model.predict([obs, action])

    def fit(self, expert_traj_s, expert_traj_a, agent_traj_s, agent_traj_a, batch_size=128, epochs=10,
            validation_split=0.15):
        # Generating the training set
        expert_label = np.ones((expert_traj_s.shape[0], 1))
        agent_label = np.zeros((agent_traj_s.shape[0], 1))

        x_s = np.concatenate([expert_traj_s, agent_traj_s], axis=0)
        x_a = np.concatenate([expert_traj_a, agent_traj_a], axis=0)

        y = np.concatenate([expert_label, agent_label], axis=0)

        self.model.fit([x_s, x_a], y, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2, validation_split=validation_split)
