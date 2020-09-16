import random
from collections import deque

import numpy as np
import tensorflow as tf
from CAPORL.RL_Agent.ActorCritic.A2C_Agent.Networks import a2c_net_continuous
from CAPORL.Memory.deque_memory import Memory
from CAPORL.RL_Agent.agent_interfaz import AgentSuper
from CAPORL.RL_Agent.ActorCritic.A2C_Agent.a2c_agent_queue_base import A2CQueueSuper


# TODO: Heredar de a2c_agent_continuous.py
def create_agent():
    return "A2C_continuous_queue"


# worker class that inits own environment, trains on it and updloads weights to global net
class Agent(A2CQueueSuper):
    def __init__(self, sess, state_size, n_actions, stack=False, img_input=False, lr_actor=0.0001, lr_critic=0.001,
                 n_steps_update=10, action_bound=None, batch_size=32, net_architecture=None):
        self.action_bound = action_bound

        super().__init__(sess, state_size, n_actions, stack=stack, img_input=img_input, lr_actor=lr_actor,
                         lr_critic=lr_critic, batch_size=batch_size, n_steps_update=n_steps_update, net_architecture=net_architecture,
                         continuous_actions=True)

        self.epsilon = 0.  # Not useful

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
        self.done = done
        self.episode_memory.append([obs, action, reward])
        self.next_obs = next_obs

    def act(self, obs):
        """
        Selecting the action using epsilon greedy policy
        :param obs: Observation (State)
        """
        """
        Selecting the action using epsilon greedy policy
        :param obs: Observation (State)
        """
        obs = self._format_obs_act(obs)
        return self.worker.choose_action(obs)

    def act_test(self, obs):
        return self.act(obs)

    def replay(self):
        """"
        Training process
        """
        self._replay()

