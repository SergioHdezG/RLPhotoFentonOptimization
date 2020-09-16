import random
import tensorflow as tf
import numpy as np
from CAPORL.RL_Agent.ActorCritic.A2C_Agent.Networks import a2c_net_discrete
from CAPORL.RL_Agent.agent_interface import AgentSuper
from CAPORL.RL_Agent.ActorCritic.A2C_Agent.a2c_agent_base import A2CSuper

def create_agent():
    return "A2C_discrete"


# worker class that inits own environment, trains on it and updloads weights to global net
class Agent(A2CSuper):
    def __init__(self, sess, state_size, n_actions, stack=False, img_input=False, lr_actor=0.0001, lr_critic=0.001,
                 epsilon=1., epsilon_decay=0.99995, epsilon_min=0.15, n_steps_update=10, batch_size=32,
                 net_architecture=None):
        super().__init__(sess, state_size, n_actions, stack=stack, img_input=img_input, lr_actor=lr_actor,
                         lr_critic=lr_critic, n_steps_update=n_steps_update, net_architecture=net_architecture,
                         continuous_actions=False)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def act(self, obs):
        """
        Selecting the action using epsilon greedy policy
        :param obs: Observation (State)
        """
        """
        Selecting the action using epsilon greedy policy
        :param obs: Observation (State)
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_actions)

        obs = self._format_obs_act(obs)
        return self.worker.choose_action(obs)

    def act_test(self, obs):
        obs = self._format_obs_act(obs)
        return self.worker.choose_action(obs)

    def remember(self, obs, action, reward, next_obs, done):
        """
        Store a memory in a list of memories
        :param obs: Current Observation (State)
        :param action: Action selected
        :param reward: Reward
        :param next_obs: Next Observation (Next State)
        :param done: If the episode is finished
        :return:
        """
        act_one_hot = np.zeros(self.n_actions)  # turn action into one-hot representation
        act_one_hot[action] = 1
        self.done = done
        self.memory.append([obs, act_one_hot, reward])
        self.next_obs = next_obs

    def replay(self):
        """"
        Training process
        """
        self._replay()
        self._reduce_epsilon()

    def _reduce_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
