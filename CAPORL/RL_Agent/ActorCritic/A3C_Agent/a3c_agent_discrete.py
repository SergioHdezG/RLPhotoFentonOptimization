import copy
import random
import gym
import numpy as np
from CAPORL.RL_Agent.ActorCritic.A3C_Agent import a3c_globals as glob
from CAPORL.RL_Agent.ActorCritic.A3C_Agent.Networks.a3c_net_discrete import *
from collections import deque
import datetime as dt
from CAPORL.utils.parse_utils import *

def create_agent():
    return "A3C_discrete"


# worker class that inits own environment, trains on it and updloads weights to global net
class Worker(object):
    def __init__(self, name, globalAC, sess, state_size, n_actions, environment, n_stack=1, img_input=False, epsilon=1.,
                 epsilon_min=0.1, epsilon_decay=0.99995, lr_actor=0.0001, lr_critic=0.001, n_steps_update=10,
                 saving_model_params=None, net_architecture=None):

        if isinstance(environment, str):
            try:
                self.env = gym.make(environment)  #.unwrapped  # make environment for each worker
            except:
                print(environment, "is not listed in gym environmets")
        else:
            try:
                self.env = environment.env()
            except:
                print("The constructor of your environment is not well defined. "
                      "To use your own environment you need a constructor like: env()")

        self.n_stack = n_stack
        self.img_input = img_input

        self.name = name
        self.n_actions = n_actions
        self.state_size = state_size

        if self.img_input:
            img_input = True
            stack = self.n_stack is not None and self.n_stack > 1
            state_size = (*self.state_size[:2], self.state_size[-1] * self.n_stack)

        elif self.n_stack is not None and self.n_stack > 1:
            img_input = False
            stack = True
            state_size = (self.state_size, self.n_stack)
        else:
            img_input = False
            stack = False
            state_size = self.state_size
        self.sess = sess
        self.saver = None  # Needs to be initialized outside this class

        self.AC = ACNet(name, self.sess, state_size, self.n_actions, stack=stack, img_input=img_input, lr_actor=lr_actor,
                        lr_critic=lr_critic, globalAC=globalAC, net_architecture=net_architecture)  # create ACNet for each worker

        self.epsilon = epsilon
        self.n_steps_update = n_steps_update
        self.gamma = 0.90
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.preprocess = None
        self.clip_reward = None

        if saving_model_params is not None:
            self.save_base, self.save_name, self.save_each, self.save_if_better = parse_saving_model_params(saving_model_params)
        else:
            self.save_base = "/home/shernandez/PycharmProjects/CAPORL_full_project/saved_models/"
            self.save_name = create_agent()
            self.save_each = None
            self.save_if_better = False

        self.max_rew_mean = -2**1000  # Store the maximum value for reward mean

    def work(self, episodes, render=False, render_after=None, max_steps_epi=None, preprocess=None, clip_reward=None,
             skip_states=1):
        self.preprocess = preprocess
        self.clip_reward = clip_reward

        # global global_rewards, global_episodes
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []

        # Stacking inputs
        if self.n_stack is not None and self.n_stack > 1:
            obs_queue = deque(maxlen=self.n_stack)
            obs_next_queue = deque(maxlen=self.n_stack)
        else:
            obs_queue = None
            obs_next_queue = None

        while not glob.coord.should_stop() and glob.global_episodes < episodes:
            obs = self.env.reset()
            ep_r = 0
            done = False
            epochs = 0

            obs = self.preprocess(obs)
            # Stacking inputs
            if self.n_stack is not None and self.n_stack > 1:
                for i in range(self.n_stack):
                    obs_queue.append(obs)
                    obs_next_queue.append(obs)

            while not done:
                if self.name == 'W_0' and (render or (render_after is not None and glob.global_episodes > render_after)):
                    self.env.render()
                action = self.act(obs, obs_queue) #self.AC.choose_action(s)  # estimate stochastic action based on policy
                next_obs, reward, done, info = self.env.step(action)  # make step in environment
                # next_obs = self.preprocess(next_obs)  # Preprocess is made now in frame_skipping function
                done, next_obs, reward, epochs = self.frame_skipping(action, done, next_obs, reward, skip_states, epochs)


                if not done and max_steps_epi is not None:
                    done = True if epochs == max_steps_epi - 1 else False

                ep_r += reward

                # stacking inputs
                if self.n_stack is not None and self.n_stack > 1:
                    obs_next_queue.append(next_obs)

                    if self.img_input:
                        obs_satck = np.dstack(obs_queue)
                        obs_next_stack = np.dstack(obs_next_queue)
                    else:
                        obs_satck = np.array(obs_queue).reshape(self.state_size, self.n_stack)
                        obs_next_stack = np.array(obs_next_queue).reshape(self.state_size, self.n_stack)

                    obs = obs_satck
                    next_obs = obs_next_stack

                # save actions, states and rewards in buffer
                buffer_s.append(obs)
                act_one_hot = np.zeros(self.n_actions)  # turn action into one-hot representation
                act_one_hot[action] = 1
                buffer_a.append(act_one_hot)
                buffer_r.append(self.clip_reward(reward))  # normalize reward

                if total_step % self.n_steps_update == 0 or done:  # update global and assign to local net
                    if done:
                        v_next_obs = 0  # terminal
                    else:
                        v_next_obs = self.sess.run(self.AC.v, {self.AC.s: next_obs[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for rew in buffer_r[::-1]:  # reverse buffer r
                        v_next_obs = rew + self.gamma * v_next_obs
                        buffer_v_target.append(v_next_obs)
                    buffer_v_target.reverse()

                    if self.img_input:
                        if self.n_stack is not None and self.n_stack > 1:
                            buffer_s = np.reshape(buffer_s, (-1, *self.state_size[:2], self.state_size[-1]*self.n_stack))
                        else:
                            buffer_s = np.reshape(buffer_s, (-1, *self.state_size[:2], self.state_size[-1]))
                    elif self.n_stack is not None and self.n_stack > 1:
                        buffer_s = np.array(buffer_s).reshape((-1, self.state_size, self.n_stack))
                    else:
                        buffer_s = np.vstack(buffer_s)

                    buffer_a, buffer_v_target = np.vstack(buffer_a), np.vstack(
                        buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)  # actual training step, update global ACNet
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()  # get global parameters to local ACNet

                    self._reduce_epsilon()

                obs, obs_queue = self.copy_next_obs(next_obs, obs, obs_next_queue, obs_queue)
                total_step += 1
                if done:
                    self._feedback_print(glob.global_episodes, ep_r, epochs, verbose=1)

                epochs += 1
        self.env.close()
        self.AC.update_global(feed_dict)  # actual training step, update global ACNet
        self.AC.pull_global()  # get global parameters to local ACNet

    def test(self, n_iter, render=True, max_steps_epi=None, preprocess=None, clip_reward=None, callback=None):
        self.preprocess = preprocess
        self.clip_reward = clip_reward

        # global global_rewards, global_episodes
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []

        # Stacking inputs
        if self.n_stack is not None and self.n_stack > 1:
            obs_queue = deque(maxlen=self.n_stack)
        else:
            obs_queue = None

        for e in range(n_iter):
            obs = self.env.reset()
            ep_r = 0
            done = False
            epochs = 0

            obs = self.preprocess(obs)
            # Stacking inputs
            if self.n_stack is not None and self.n_stack > 1:
                for i in range(self.n_stack):
                    obs_queue.append(obs)

            while not done:
                if render:
                    self.env.render()

                a = self.act_test(obs, obs_queue) #self.AC.choose_action(s)  # estimate stochastic action based on policy
                prev_obs = obs

                obs, r, done, info = self.env.step(a)  # make step in environment
                obs = self.preprocess(obs)

                if callback is not None:
                    callback(prev_obs, obs, a, r, done, info)


                # if not done and max_steps_epi is not None:
                #     done = True if epochs == max_steps_epi - 1 else False

                # stacking inputs
                if self.n_stack is not None and self.n_stack > 1:
                    obs_queue.append(obs)
                epochs += 1
                ep_r += r
            total_step += 1
            if done:
                self._feedback_print(glob.global_episodes, ep_r, epochs, verbose=1)
        self.env.close()


    def act(self, obs, obs_queue):
        """
        Selecting the action using epsilon greedy policy
        :param obs: Observation (State)
        :return:
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_actions)  # '''<-- Exploration'''

        if self.img_input:
            if self.n_stack is not None and self.n_stack > 1:
                return self.AC.choose_action(np.array(obs_queue).reshape(-1, *self.state_size[:2], self.state_size[-1] * self.n_stack))
            else:
                return self.AC.choose_action(np.array(obs).reshape(-1, *self.state_size[:2], self.state_size[-1]))
        if self.n_stack is not None and self.n_stack > 1:
            return self.AC.choose_action(np.array(obs_queue).reshape(-1, self.state_size, self.n_stack))
        else:
            return self.AC.choose_action(np.array(obs).reshape(-1,self.state_size))  # '''<-- Exploitation'''

    def act_test(self, obs, obs_queue):
        """
        Selecting the action using epsilon greedy policy
        :param obs: Observation (State)
        :return:
        """
        if self.img_input:
            if self.n_stack is not None and self.n_stack > 1:
                return self.AC.choose_action(np.array(obs_queue).reshape(-1, *self.state_size[:2], self.state_size[-1] * self.n_stack))
            else:
                return self.AC.choose_action(np.array(obs).reshape(-1, *self.state_size[:2], self.state_size[-1]))
        if self.n_stack is not None and self.n_stack > 1:
            return self.AC.choose_action(np.array(obs_queue).reshape(-1, self.state_size, self.n_stack))
        else:
            return self.AC.choose_action(np.array(obs).reshape(-1,self.state_size))  # '''<-- Exploitation'''

    def _feedback_print(self, e, episodic_reward, epochs, verbose):
        glob.global_raw_rewards.append(episodic_reward)
        # if len(glob.global_rewards) < 10:  # record running episode reward
        #     glob.global_rewards.append(episodic_reward)
        #     smooth_rew = glob.global_rewards[-1]
        # else:
        #     glob.global_rewards.append(episodic_reward)
        #     reward_mean = glob.global_rewards[-1]
        #     smooth_rew = (np.mean(glob.global_rewards[-5:]))
        #     glob.global_rewards[-1] = (np.mean(glob.global_rewards[-5:]))  # smoothing

        rew_mean = np.sum(glob.global_raw_rewards) / len(glob.global_raw_rewards)
        if verbose == 1:
            if (e + 1) % 1 == 0:
                print('Episode ', e + 1, 'Epochs ', epochs, ' Reward {:.1f}'.format(episodic_reward),
                      'Smooth Reward {:.1f}'.format(rew_mean), ' Epsilon {:.4f}'.format(self.epsilon))

            if self.save_each is not None and (e + 1) % self.save_each == 0:
                print(dt.datetime.now())
                if self._check_for_save(rew_mean):
                    self.save(self.save_base + self.save_name, int(rew_mean))

        if verbose == 2:
            print('Episode ', e + 1, 'Mean Reward ', rew_mean)

        glob.global_episodes += 1

    def preprocess(self, obs):
        return obs

    def clip_reward(self, obs):
        return obs

    def copy_next_obs(self, next_obs, obs, obs_next_queue, obs_queue):
        if self.n_stack is not None and self.n_stack > 1:
            obs_queue = copy.copy(obs_next_queue)
        else:
            obs = next_obs
        return obs, obs_queue

    def _reduce_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, dir, name):
        self.worker.load(dir, name)

    def save(self, name, reward):
        self.saver.save(self.sess, name, global_step=reward)
        print("Model saved to disk")

    def _check_for_save(self, rew_mean):
        if self.save_if_better:
            if rew_mean > self.max_rew_mean:
                self.max_rew_mean = rew_mean
                return True
            else:
                return False
        return True

    def frame_skipping(self, action, done, next_obs, reward, skip_states, epochs):
        if skip_states > 1 and not done:
            for i in range(skip_states - 2):
                next_obs_aux1, reward_aux, done_aux, _ = self.env.step(action)
                epochs += 1
                reward += reward_aux
                if done_aux:
                    next_obs_aux2 = next_obs_aux1
                    done = done_aux
                    break

            if not done:
                next_obs_aux2, reward_aux, done_aux, _ = self.env.step(action)
                epochs += 1
                reward += reward_aux
                done = done_aux

            if self.img_input:
                next_obs_aux2 = self.preprocess(next_obs_aux2)
                if skip_states > 2:
                    next_obs_aux1 = self.preprocess(next_obs_aux1)
                    next_obs = np.maximum(next_obs_aux2, next_obs_aux1)
                else:
                    next_obs = self.preprocess(next_obs)
                    next_obs = np.maximum(next_obs_aux2, next_obs)
            else:
                next_obs = self.preprocess(next_obs_aux2)
        else:
            next_obs = self.preprocess(next_obs)
        return done, next_obs, reward, epochs