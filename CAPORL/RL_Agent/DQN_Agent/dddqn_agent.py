# -*- coding: utf-8 -*-
import numpy as np

'''Un objeto deque es un contenedor de datos del módulo collections
   similar a una lista o una cola 
   que permite añadir o suprimir elementos por sus dos extremos. '''
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda, subtract, add
from tensorflow.keras.optimizers import Adam
from CAPORL.RL_Agent.DQN_Agent.dqn_agent_base import DQNAgentSuper
from CAPORL.utils import net_building
from CAPORL.utils.networks import dddqn_net


def create_agent():
    return 'DDDQN'

class Agent(DQNAgentSuper):
    """
    Dueling (Double) Deep Q Network Agent extend DQNAgentSuper
    """
    def __init__(self, n_actions, state_size=4, batch_size=32, epsilon_min=0.01, epsilon_decay=0.9999995,
                 learning_rate=1e-4, gamma=0.95, epsilon=.8, stack=False, img_input=False,
                 model_params=None, net_architecture=None):
        """
        Attributes:
            n_actions:          Int. Number of different actions.
            state_size:         Int or Tuple. State dimensions.
            batch_size:         Int. Batch size for training.
            epsilon_min:        Min value epsilon can take.
            epsilon_decay:      Decay rate for epsilon.
            learning_rate:      Learning rate for training.
            gamma:              Discount factor for target value.
            epsilon:            Initial value for epsilon.
            stack:              True if stacked inputs are used, False otherwise.
            img_input:          True if inputs are images, False otherwise.
            model_params:       Dictionary of params like learning rate, batch size, epsilon values, n step returns...
        """
        super().__init__(n_actions, state_size=state_size, batch_size=batch_size, epsilon_min=epsilon_min,
                         epsilon_decay=epsilon_decay, learning_rate=learning_rate,
                         gamma=gamma, epsilon=epsilon, stack=stack, img_input=img_input, model_params=model_params,
                         net_architecture=net_architecture)

    def _build_model(self, net_architecture):
        # Neural Net for Deep-Q learning Model
        if net_architecture is None:  # Standart architecture
            net_architecture = dddqn_net

        # model = Sequential()
        if self.img_input:  # and (self.stack ot not self.stack)
            # model.add(Conv2D(32, input_shape=self.state_size, kernel_size=9, strides=(4, 4), padding='same', activation='relu', name="conv_1"))
            # model.add(Conv2D(64, kernel_size=5, strides=(2, 2), padding='same', activation='relu', name="conv_2"))
            # model.add(Conv2D(64, kernel_size=3, strides=(2, 2), padding='same', activation='relu', name="conv_3"))
            # model.add(Flatten())
            model, dense_v, dense_a = net_building.build_conv_net(net_architecture, self.state_size, dddqn=True)
        elif self.stack:
            # model.add(Flatten(input_shape=self.state_size))
            # model.add(Dense(64, activation='relu', name="dense_1"))
            model, dense_v, dense_a = net_building.build_stack_net(net_architecture, self.state_size, dddqn=True)

        else:
            # model.add(Dense(256, input_dim=self.state_size, activation='relu', name="dense_1"))
            model, dense_v, dense_a = net_building.build_nn_net(net_architecture, self.state_size, dddqn=True)

        # Value model
        # dense_v = Dense(256, activation='relu', name="dense_valor")(model.output)
        out_v = Dense(1, activation='linear', name="out_valor")(dense_v)

        # Advantage model
        # dense_a = Dense(256, activation='relu', name="dense_advantage")(model.output)
        out_a = Dense(self.action_size, activation='linear', name="out_advantage")(dense_a)

        a_mean = Lambda(tf.math.reduce_mean, arguments={'axis': 1, 'keepdims': True})(out_a)  # K.mean
        a_subs = subtract([out_a, a_mean])
        output = add([out_v, a_subs])
        # output = add([out_v, out_a])
        duelingDQN = Model(inputs=model.inputs, outputs=output)
        duelingDQN.compile(loss='mse',
                           optimizer=Adam(lr=self.learning_rate))

        return duelingDQN

    def _calc_target(self, done, reward, next_obs):
        """
        Calculate the target values for matching the DQN training process.
        """
        armax = np.argmax(self.model.predict(next_obs), axis=1)
        target_value = self.target_model.predict(next_obs)
        values = []

        for i in range(target_value.shape[0]):
            values.append(target_value[i][armax[i]])

        # l = np.amax(self.target_model.predict(next_obs), axis=1)
        target_aux = (reward + self.gamma * np.array(values))
        target = reward

        not_done = [not i for i in done]
        target__aux = target_aux * not_done
        target = done * target

        return target__aux + target
