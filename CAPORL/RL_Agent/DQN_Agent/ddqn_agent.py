# -*- coding: utf-8 -*-
import numpy as np

'''Un objeto deque es un contenedor de datos del módulo collections
   similar a una lista o una cola 
   que permite añadir o suprimir elementos por sus dos extremos. '''
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from CAPORL.RL_Agent.base.DQN_base.dqn_agent_base import DQNAgentSuper
from CAPORL.utils import net_building
from CAPORL.utils.networks import ddqn_net
from CAPORL.RL_Agent.base.utils import agent_globals


def create_agent():
    return 'DDQN'

class Agent(DQNAgentSuper):
    """
    Double Deep Q Network Agent extend DQNAgentSuper
    """
    def __init__(self):
        self.agent_name = agent_globals.names["ddqn"]

    def build_agent(self, n_actions, state_size=4, batch_size=32, epsilon_min=0.1, epsilon_decay=0.999995,
                 learning_rate=1e-3, gamma=0.95, epsilon=1., stack=False, img_input=False,
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
                         epsilon_decay=epsilon_decay, learning_rate=learning_rate, gamma=gamma, epsilon=epsilon,
                         stack=stack, img_input=img_input, model_params=model_params, net_architecture=net_architecture)

    def _build_model(self, net_architecture):
        # Neural Net for Deep-Q learning Model
        if net_architecture is None:  # Standart architecture
            net_architecture = ddqn_net

        if self.img_input:
            model = net_building.build_conv_net(net_architecture, self.state_size)

        elif self.stack:
            model = net_building.build_stack_net(net_architecture, self.state_size)
        else:
            model = net_building.build_nn_net(net_architecture, self.state_size)
        model.add(Dense(self.n_actions, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

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