import threading
import gym
import matplotlib.pyplot as plt
import tensorflow as tf
import os.path as path
import multiprocessing
from _collections import deque
from CAPORL.RL_Agent.ActorCritic.A3C_Agent import a3c_agent_continuous, a3c_agent_discrete, \
    a3c_globals as glob
from CAPORL.RL_Agent.ActorCritic.A3C_Agent.Networks import a3c_net_continuous, a3c_net_discrete
from CAPORL.utils.parse_utils import *

class A3CProblem:
    """
    Asynchronous Advantage Actor-Critic.
    This algorithm is the only one whitch does not extend RLProblemSuper because it has a different architecture.
    """
    def __init__(self, environment, agent, n_stack=1, img_input=False, state_size=None, model_params=None,
                 saving_model_params=None, net_architecture=None):
        """
        Attributes:
                environment:    Environment selected for this problem
                agent:          Agent to solve the problem: DDPG.
                n_stack:        Int >= 1. If 1, there is no stacked input. Number of time related input stacked.
                img_input:      Bool. If True, input data is an image.
                state_size:     None, Int or Tuple. State dimensions. If None it will be calculated automatically. Int
                                or Tuple format will be useful when preprocessing change the input dimensions.
                model_params:   Dictionary of params like learning rate, batch size, epsilon values, n step returns...
        """
        self.environment = environment

        if isinstance(self.environment, str):
            try:
                env = gym.make(self.environment)
            except:
                print(environment, "is not listed in gym environmets")
        else:
            try:
                env = self.environment.env()
            except:
                print("The constructor of your environment is not well defined. "
                      "To use your own environment you need a constructor like: env()")

        env.reset()

        if model_params is not None:
            batch_size, epsilon, epsilon_min, epsilon_decay, learning_rate, n_step_rew = \
                parse_model_params(model_params)

        if state_size is None:
            if img_input:
                self.state_size = env.observation_space.shape
            else:
                self.state_size = env.observation_space.shape[0]
        else:
            self.state_size = state_size

        # Set n_actions depending on the enviroment format
        try:
            self.n_actions = env.action_space.n
        except AttributeError:
            self.n_actions = env.action_space.shape[0]

        self.n_stack = n_stack
        self.img_input = img_input

        self.discrete = agent == "A3C_discrete"
        if self.discrete:

            self.ACNet = a3c_net_discrete.ACNet
            self.Worker = a3c_agent_discrete.Worker
        else:
            self.ACNet = a3c_net_continuous.ACNet
            self.Worker = a3c_agent_continuous.Worker
            self.action_bound = [env.action_space.low, env.action_space.high]  # action bounds

        self.global_net_scope = "Global_Net"
        self.n_workers = multiprocessing.cpu_count()  # number of workers
        self.n_steps_update = n_step_rew
        self.lr_actor = learning_rate*0.1
        self.lr_critic = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.sess = tf.Session()
        # self.saver = None  # Needs to be initializated after building the model
        self._build_agent(saving_model_params, net_architecture)
        self.sess.run(tf.global_variables_initializer())

        self.preprocess = self._preprocess
        self.clip_reward = self._clip_reward

    def solve(self, episodes, verbose=1, render=False, render_after=None, max_step_epi=None, skip_states=1):

        glob.global_raw_rewards = deque(maxlen=100)
        glob.global_episodes = 0

        glob.coord = tf.train.Coordinator()

        worker_threads = []
        for worker in self.workers:  # start workers
            job = lambda: worker.work(episodes, render, render_after, max_step_epi, self.preprocess, self.clip_reward,
                                      skip_states=skip_states)
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)
        glob.coord.join(worker_threads)  # wait for termination of workers

    def test(self, n_iter=10, render=True, callback=None):
        """ Test a trained agent on an environment

        :param n_iter: int. number of test iterations
        :param name_loaded: string. Name of file of model to load. If empty, no model will be loaded.
        :param render: bool. Render or not
        :return:
        """
        glob.global_raw_rewards = deque(maxlen=100)
        glob.global_episodes = 0

        self.workers[-1].test(n_iter, render, self.preprocess, self.clip_reward)


    def _build_agent(self, saving_model_params, net_architecture):
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

        with tf.device("/cpu:0"):
            if self.discrete:
                self.global_ac = self.ACNet(self.global_net_scope, self.sess, state_size, self.n_actions,
                                            stack=stack, img_input=img_input, lr_actor=self.lr_actor,
                                            lr_critic=self.lr_critic, net_architecture=net_architecture)  # we only need its params
            else:
                self.global_ac = self.ACNet(self.global_net_scope, self.sess, state_size, self.n_actions,
                                            lr_actor=self.lr_actor, lr_critic=self.lr_critic, stack=stack,
                                            img_input=img_input, action_bound=self.action_bound,
                                            net_architecture=net_architecture)  # we only need its params
            self.saver = tf.train.Saver()
            self.workers = []
            # Create workers
            for i in range(self.n_workers):
                i_name = 'W_%i' % i  # worker name
                if self.discrete:
                    self.workers.append(
                        self.Worker(i_name, self.global_ac, self.sess, self.state_size, self.n_actions, self.environment,
                                    n_stack=self.n_stack, img_input=self.img_input, epsilon=self.epsilon,
                                    epsilon_min=self.epsilon_min, epsilon_decay=self.epsilon_decay,
                                    lr_actor=self.lr_actor, lr_critic=self.lr_critic,
                                    n_steps_update=self.n_steps_update, saving_model_params=saving_model_params))
                else:
                    self.workers.append(
                        self.Worker(i_name, self.global_ac, self.sess, self.state_size, self.n_actions, self.environment,
                                    n_stack=self.n_stack, img_input=self.img_input, lr_actor=self.lr_actor,
                                    lr_critic=self.lr_critic, n_steps_update=self.n_steps_update,
                                    action_bound=self.action_bound, saving_model_params=saving_model_params))


        for w in self.workers:
            w.saver = self.saver

    def _preprocess(self, obs):
        return obs

    def _clip_reward(self, obs):
        return obs

    def load_model(self, dir_load="", name_loaded=""):
        self.load(dir_load, name_loaded)

    def load(self, dir, name):
        name = path.join(dir, name)
        loaded_model = tf.train.import_meta_graph(name + '.meta')
        loaded_model.restore(self.sess, tf.train.latest_checkpoint(dir + "./"))
        for worker in self.workers:
            worker.AC.pull_global()  # get global parameters to local ACNet
