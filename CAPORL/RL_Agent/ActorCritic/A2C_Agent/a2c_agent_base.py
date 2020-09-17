from os import path
import datetime
import random
from tensorflow.python.keras.models import model_from_json
import random
from  CAPORL.RL_Agent.agent_interface import AgentInterface
import numpy as np
import tensorflow as tf
from CAPORL.RL_Agent.ActorCritic.A2C_Agent.Networks import a2c_net_discrete
from CAPORL.RL_Agent.ActorCritic.A2C_Agent.Networks import a2c_net_continuous
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from CAPORL.utils import net_building
from CAPORL.utils.networks import ppo_net
from tensorflow.keras.initializers import RandomNormal
from CAPORL.RL_Agent.agent_interface import AgentSuper


# worker class that inits own environment, trains on it and updloads weights to global net
class A2CSuper(AgentSuper):
    def __init__(self, sess, state_size, n_actions, stack=False, img_input=False, lr_actor=0.0001, lr_critic=0.001,
                 n_steps_update=10, net_architecture=None, continuous_actions=False):
        super().__init__(state_size=state_size, n_actions=n_actions, img_input=img_input, stack=stack)

        self.sess = sess

        self.n_steps_update = n_steps_update
        self.gamma = 0.90
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self._build_net(net_architecture, continuous=continuous_actions)
        self.saver = tf.train.Saver()

        self.memory = []
        self.done = False
        self.total_step = 1
        self.next_obs = None

    def _build_net(self, net_architecture, continuous=False):
        if not continuous:
            ACNet = a2c_net_discrete.ACNet
            self.worker = ACNet("Worker", self.sess, self.state_size, self.n_actions, stack=self.stack,
                                img_input=self.img_input, lr_actor=self.lr_actor,
                                lr_critic=self.lr_critic,
                                net_architecture=net_architecture)  # create ACNet for each worker
        else:
            with tf.device("/cpu:0"):
                ACNet = a2c_net_continuous.ACNet
                self.worker = ACNet("Worker", self.sess, self.state_size, self.n_actions, stack=self.stack,
                                    img_input=self.img_input, lr_actor=self.lr_actor, lr_critic=self.lr_critic,
                                    action_bound=self.action_bound,
                                    net_architecture=net_architecture)  # create ACNet for each worker

    def load_memories(self):
        """
        Load a batch of memories
        :return: Current Observation, Action, Reward, Next Observation, episode finished flag
        """
        memory = np.array(self.memory)
        obs, action, reward = memory[:, 0], memory[:, 1], memory[:, 2]
        obs = np.array([x.reshape(self.state_size) for x in obs])
        action = np.array([x.reshape(self.n_actions) for x in action])
        self.memory = []
        return obs, action, reward

    def _replay(self):
        """"
        Training process
        """
        if self.total_step % self.n_steps_update == 0 or self.done:  # update global and assign to local net
            obs_buff, actions_buff, reward_buff = self.load_memories()

            if self.done:
                v_next_obs = 0  # terminal
            else:
                v_next_obs = self.sess.run(self.worker.v, {self.worker.s: self.next_obs[np.newaxis, :]})[0, 0]
            buffer_v_target = []

            for r in reward_buff[::-1]:  # reverse buffer r
                v_next_obs = r + self.gamma * v_next_obs
                buffer_v_target.append(v_next_obs)
            buffer_v_target.reverse()
            buffer_v_target = np.vstack(buffer_v_target)

            feed_dict = {
                self.worker.s: obs_buff,
                self.worker.a_his: actions_buff,
                self.worker.v_target: buffer_v_target,
            }
            self.worker.update_global(feed_dict)  # actual training step, update global ACNet

        self.total_step += 1

    def load(self, dir, name):
        self.worker.load(dir, name)

    def save(self, name, reward):
        self.saver.save(self.sess, name, global_step=reward)
        print("Model saved to disk")