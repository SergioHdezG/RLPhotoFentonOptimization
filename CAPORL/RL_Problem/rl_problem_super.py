import copy
import gc

import gym
import numpy as np
import datetime as dt
from collections import deque
import matplotlib.pyplot as plt
# from pympler import muppy, summary
# from memory_leaks import *
import tensorflow.keras.backend as K
from CAPORL.utils.parse_utils import *

class RLProblemSuper:
    """ Reinforcement Learning Problem.

    This class represent the RL problem to solve, here is where the problem gave by the environment is solved by the
    agent.
    All agents extends this class,but A3C due to a different structure is needed in this algorithm.
    """

    def __init__(self, environment, agent, n_stack=1, img_input=False, state_size=None, saving_model_params=None,
                 net_architecture=None):
        """
        Attributes:
            environment:    Environment selected for this problem
            agent:          Agent to solve the problem
            n_stack:        Int >= 1. If 1, there is no stacked input. Number of time related input stacked.
            img_input:      Bool. If True, input data is an image.
            state_size:     None, Int or Tuple. State dimensions. If None it will be calculated automatically. Int or
                            Tuple format will be useful when preprocessing change the input dimensions.
        """

        # Inicializar el entorno
        if isinstance(environment, str):
            self.env = gym.make(environment)
        else:
            self.env = environment()

        self.n_stack = n_stack
        self.img_input = img_input

        # Set state_size depending on the input type
        if state_size is None:
            if img_input:
                self.state_size = self.env.observation_space.shape
            else:
                self.state_size = self.env.observation_space.shape[0]
        else:
            self.state_size = state_size

        # Set n_actions depending on the environment format
        try:
            self.n_actions = self.env.action_space.n
        except AttributeError:
            self.n_actions = self.env.action_space.shape[0]

        # Setting default preprocess and clip_norm_reward functions
        self.preprocess = self._preprocess  # Preprocessing function for observations
        self.clip_norm_reward = self._clip_norm_reward  # Clipping reward

        # The agent will be initialized in agent subclass
        self.agent = None

        # Total number of steps processed
        self.global_steps = 0

        if saving_model_params is not None:
            self.save_base, self.save_name, self.save_each, self.save_if_better = parse_saving_model_params(saving_model_params)
        else:
            self.save_base = self.save_name = self.save_each = self.save_if_better = None

        self.max_rew_mean = -2**1000  # Store the maximum value for reward mean

    def _build_agent(self, agent, model_params, net_architecture):
        # Abstract function
        pass

    def solve(self, episodes, render=True, render_after=None, max_step_epi=None, skip_states=1, verbose=1, discriminator=None):
        """ Algorithm for training the agent to solve the environment problem.

        :param episodes:        Int >= 1. Number of episodes to train.
        :param render:          Bool. If True, the environment will show the user interface during the training process.
        :param render_after:    Int >=1 or None. Star rendering the environment after this number of episodes.
        :param max_step_epi:    Int >=1 or None. Maximum number of epochs per episode. Mainly for problems where the
                                environment
                                doesn't have a maximum number of epochs specified.
        :param skip_states:     Int >= 1. Frame skipping technique  applied in Playing Atari With Deep Reinforcement
                                Learning paper. If 1, this technique won't be applied.
        :param verbose:         Int in range [0, 2]. If 0 no training information will be displayed, if 1 lots of
                                information will be displayed, if 2 fewer information will be displayed.
        :return:
        """
        # Inicializar iteraciones globales
        self.global_steps = 0

        # List of 100 last rewards
        rew_mean_list = deque(maxlen=10)

        # Stacking inputs
        if self.n_stack is not None and self.n_stack > 1:
            obs_queue = deque(maxlen=self.n_stack)
            obs_next_queue = deque(maxlen=self.n_stack)
        else:
            obs_queue = None
            obs_next_queue = None

        # For each episode do
        for e in range(episodes):

            # Init episode parameters
            obs = self.env.reset()
            episodic_reward = 0
            epochs = 0
            done = False

            # Reading initial state
            obs = self.preprocess(obs)
            # obs = np.zeros((300, 300))

            # Stacking inputs
            if self.n_stack is not None and self.n_stack > 1:
                for i in range(self.n_stack):
                    obs_queue.append(np.zeros(obs.shape))
                    obs_next_queue.append(np.zeros(obs.shape))
                obs_queue.append(obs)
                obs_next_queue.append(obs)

            # While the episode doesn't reach a final state
            while not done:
                if render or ((render_after is not None) and e > render_after):
                    self.env.render()

                # Select an action
                action = self.act(obs, obs_queue)

                # Agent act in the environment
                next_obs, reward, done, _ = self.env.step(action)
                if discriminator is not None:
                    reward = discriminator.get_reward(obs, action)[0]
                # next_obs = np.zeros((300, 300))
                # next_obs = self.preprocess(next_obs)  # Is made in store_experience now

                # Store the experience in memory
                next_obs, obs_next_queue, reward, done, epochs = self.store_experience(action, done, next_obs, obs, obs_next_queue,
                                                                         obs_queue, reward, skip_states, epochs)

                # Replay some memories and training the agent
                self.agent.replay()

                # copy next_obs to obs
                obs, obs_queue = self.copy_next_obs(next_obs, obs, obs_next_queue, obs_queue)

                # If max steps value is reached the episode is finished
                done = self._max_steps(done, epochs, max_step_epi)

                episodic_reward += reward
                epochs += 1
                self.global_steps += 1

            # Add reward to the list
            rew_mean_list.append(episodic_reward)

            # Copy target Q model to main Q model
            self.agent.copy_model_to_target()

            # Print log on scream
            self._feedback_print(e, episodic_reward, epochs, verbose, rew_mean_list)
        return

    def copy_next_obs(self, next_obs, obs, obs_next_queue, obs_queue):
        if self.n_stack is not None and self.n_stack > 1:
            obs_queue = copy.copy(obs_next_queue)
        else:
            obs = next_obs
        return obs, obs_queue

    def act(self, obs, obs_queue):
        # Select an action depending on stacked input or not
        if self.n_stack is not None and self.n_stack > 1:
            action = self.agent.act(np.array(obs_queue))
        else:
            action = self.agent.act(obs)
        return action

    def act_test(self, obs, obs_queue):
        # Select an action in testing mode
        if self.n_stack is not None and self.n_stack > 1:
            action = self.agent.act_test(np.array(obs_queue))
        else:
            action = self.agent.act_test(obs)
        return action

    def store_experience(self, action, done, next_obs, obs, obs_next_queue, obs_queue, reward, skip_states, epochs):

        done, next_obs, reward, epochs = self.frame_skipping(action, done, next_obs, reward, skip_states, epochs)

        # Store the experience in memory depending on stacked inputs and observations type
        if self.n_stack is not None and self.n_stack > 1:
            obs_next_queue.append(next_obs)

            if self.img_input:
                obs_satck = np.dstack(obs_queue)
                obs_next_stack = np.dstack(obs_next_queue)
            else:
                obs_satck = np.array(obs_queue).reshape(self.state_size, self.n_stack)
                obs_next_stack = np.array(obs_next_queue).reshape(self.state_size, self.n_stack)
            self.agent.remember(obs_satck, action, self.clip_norm_reward(reward), obs_next_stack, done)
        else:
            self.agent.remember(obs, action, self.clip_norm_reward(reward), next_obs, done)
        return next_obs, obs_next_queue, reward, done, epochs

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

    def test(self, n_iter=10, render=True, callback=None):
        """ Test a trained agent on an environment

        :param n_iter: int. number of test iterations
        :param name_loaded: string. Name of file of model to load. If empty, no model will be loaded.
        :param render: bool. Render or not
        :return:
        """
        epi_rew_mean = 0
        rew_mean_list = deque(maxlen=10)

        # Stacking inputs
        if self.n_stack is not None and self.n_stack > 1:
            obs_queue = deque(maxlen=self.n_stack)
        else:
            obs_queue = None

        # For each episode do
        for e in range(n_iter):
            done = False
            episodic_reward = 0
            epochs = 0
            obs = self.env.reset()
            obs = self.preprocess(obs)

            # stacking inputs
            if self.n_stack is not None and self.n_stack > 1:
                for i in range(self.n_stack):
                    obs_queue.append(np.zeros(obs.shape))
                obs_queue.append(obs)

            while not done:
                if render:
                    self.env.render()

                # Select action
                action = self.act_test(obs, obs_queue)
                prev_obs = obs

                obs, reward, done, _ = self.env.step(action)
                obs = self.preprocess(obs)

                if callback is not None:
                    callback(prev_obs, obs, action, reward, done, _)

                episodic_reward += reward
                epochs += 1

                if self.n_stack is not None and self.n_stack > 1:
                    obs_queue.append(obs)

            rew_mean_list.append(episodic_reward)

            self._feedback_print(e, episodic_reward, epochs, 1, rew_mean_list)

        # print('Mean Reward ', epi_rew_mean / n_iter)
        self.env.close()
        return

    def _preprocess(self, obs):
        return obs

    def _clip_norm_reward(self, rew):
        return rew

    def _max_steps(self, done, epochs, max_steps):
        """ Return True if epochs pass a selected number of steps"""
        if max_steps is not None:
            return epochs >= max_steps or done
        return done

    def _feedback_print(self, e, episodic_reward, epochs, verbose, epi_rew_list):
        rew_mean = np.sum(epi_rew_list) / len(epi_rew_list)

        if verbose == 1:
            if (e + 1) % 1 == 0:
                print('Episode ', e + 1, 'Epochs ', epochs, ' Reward {:.1f}'.format(episodic_reward),
                      'Smooth Reward {:.1f}'.format(rew_mean), ' Epsilon {:.4f}'.format(self.agent.epsilon))

            if self.save_each is not None and (e + 1) % self.save_each == 0:

                # print('Memory agent len: ', self.agent.memory.len())
                # print('Memory inputs: ', self.global_steps)
                print(dt.datetime.now())
                if self._check_for_save(rew_mean):
                    self.agent.save(self.save_base + self.save_name + str(int(rew_mean)), e)

                # gc.collect()
        if verbose == 2:
            print('Episode ', e + 1, 'Mean Reward ', rew_mean)

    def _check_for_save(self, rew_mean):
        if self.save_if_better:
            if rew_mean > self.max_rew_mean:
                self.max_rew_mean = rew_mean
                return True
            else:
                return False
        return True

    def load_model(self, dir_load="", name_loaded=""):
        self.agent.load(dir_load, name_loaded)
