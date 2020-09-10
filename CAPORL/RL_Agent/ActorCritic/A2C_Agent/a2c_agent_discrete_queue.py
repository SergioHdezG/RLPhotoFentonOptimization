import random
import tensorflow as tf
from collections import deque
import gc

import numpy as np
from CAPORL.RL_Agent.ActorCritic.A2C_Agent.Networks import a2c_net_discrete
from CAPORL.Memory.deque_memory import Memory

# TODO: Heredar de a2c_agent_discrete.py
def create_agent():
    return "A2C_discrete_queue"


# worker class that inits own environment, trains on it and updloads weights to global net
class Agent(object):
    def __init__(self, sess, state_size, n_actions, stack=False, img_input=False, lr_actor=0.0001, lr_critic=0.001,
                 epsilon=1., epsilon_decay=0.99995, epsilon_min=0.15, n_steps_update=10, batch_size=32,
                 net_architecture=None):

        self.batch_size = batch_size
        self.stack = stack
        self.img_input = img_input

        self.n_actions = n_actions
        self.state_size = state_size
        self.sess = sess

        self.n_steps_update = n_steps_update
        self.gamma = 0.90
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self._build_net(net_architecture)
        self.saver = tf.train.Saver()

        self.memory = Memory(maxlen=10000)
        self.episode_memory = []
        self.done = False
        self.total_step = 1
        self.next_obs = None

    def _build_net(self, net_architecture):
        ACNet = a2c_net_discrete.ACNet
        self.worker = ACNet("Worker", self.sess, self.state_size, self.n_actions, stack=self.stack,
                            img_input=self.img_input, lr_actor=self.lr_actor, lr_critic=self.lr_critic,
                            net_architecture=net_architecture)
                            # globalAC=self.global_ac)  # create ACNet for each worker

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
        self.episode_memory.append([obs, act_one_hot, reward])
        self.next_obs = next_obs

    def remember_episode(self, obs, action, v_target):
        """
        Store a memory in a list of memories
        :param obs: Current Observation (State)
        :param action: Action selected
        :param reward: Reward
        :param next_obs: Next Observation (Next State)
        :param done: If the episode is finished
        :return:
        """
        for o, a, v in zip(obs, action, v_target):
            self.memory.append([o, a, v])


    def act(self, obs):
        """
        Selecting the action using epsilon greedy policy
        :param obs: Observation (State)
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_actions)

        if self.img_input:
            obs = np.squeeze(obs, axis=3)
            obs = obs.transpose(1, 2, 0)
            obs = np.array([obs])

        elif self.stack:
            obs = np.array([obs])
        else:
            obs = obs.reshape(-1, self.state_size)

        return self.worker.choose_action(obs)

    def act_test(self, obs):
        if self.img_input:
            obs = np.squeeze(obs, axis=3)
            obs = obs.transpose(1, 2, 0)
            obs = np.array([obs])

        elif self.stack:
            obs = np.array([obs])
        else:
            obs = obs.reshape(-1, self.state_size)

        return self.worker.choose_action(obs)

    def load_main_memories(self):
        """
        Load a batch of memories
        :return: Current Observation, Action, Reward, Next Observation, episode finished flag
        """
        _, memory, _ = self.memory.sample(self.batch_size)
        # memory = np.array(random.sample(self.memory, self.batch_size))
        obs, action, v_target = memory[:, 0], memory[:, 1], memory[:, 2]
        obs = np.array([x.reshape(self.state_size) for x in obs])
        action = np.array([x.reshape(self.n_actions) for x in action])
        v_target = np.vstack(v_target)
        return obs, action, v_target

    def load_episode_memories(self):
        """
        Load a batch of memories
        :return: Current Observation, Action, Reward, Next Observation, episode finished flag
        """
        episode_memory = np.array(self.episode_memory)
        obs, action, reward = episode_memory[:, 0], episode_memory[:, 1], episode_memory[:, 2]
        # obs = np.array([x.reshape(self.state_size) for x in obs])
        # action = np.array([x.reshape(self.n_actions) for x in action])
        self.episode_memory.clear()

        return obs, action, reward


    def replay(self):
        """"
        Training process
        """
        if self.memory.len() > self.batch_size:
            obs_buff, actions_buff, buffer_v_target = self.load_main_memories()
            feed_dict = {
                self.worker.s: obs_buff,
                self.worker.a_his: actions_buff,
                self.worker.v_target: buffer_v_target,
            }
            self.worker.update_global(feed_dict)  # actual training step, update global ACNet

        if self.total_step % self.n_steps_update == 0 or self.done:  # update global and assign to local net
            obs_buff, actions_buff, reward_buff = self.load_episode_memories()

            if self.done:
                v_next_obs = 0  # terminal
                gc.collect()  # Force garbage collector to free memory
            else:
                v_next_obs = self.sess.run(self.worker.v, {self.worker.s: self.next_obs[np.newaxis, :]})[0, 0]

            buffer_v_target = []

            for r in reward_buff[::-1]:  # reverse buffer r
                v_next_obs = r + self.gamma * v_next_obs
                buffer_v_target.append(v_next_obs)
            buffer_v_target.reverse()

            self.remember_episode(obs_buff, actions_buff, buffer_v_target)

        self.total_step += 1
        self._reduce_epsilon()

    def copy_model_to_target(self):
        pass

    def load(self, dir, name):
        self.worker.load(dir, name)

    def save(self, name, reward):
        self.saver.save(self.sess, name, global_step=reward)
        print("Saved model to disk")

    def _reduce_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def set_memory(self, memory, maxlen):
        self.memory = memory(maxlen=maxlen)