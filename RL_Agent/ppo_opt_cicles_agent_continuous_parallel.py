import numpy as np
from RL_Agent.base.PPO_base.ppo_agent_base import PPOSuper
from RL_Agent.base.utils import agent_globals
import multiprocessing
from scipy.optimize import minimize
from collections import deque
from src.environments import perox_sqp_globals


def create_agent():
    return agent_globals.names["ppooptcicles_continuous_parallel_v2"]

class Agent(PPOSuper):
    def __init__(self, opt_iters=5, opt_maxiter=40, actor_lr=1e-4, critic_lr=1e-3, batch_size=32, epsilon=1.0, epsilon_decay=1., epsilon_min=0.1,
                 gamma=0.95, n_step_return=10, memory_size=512, loss_clipping=0.2, loss_critic_discount=0.5,
                 loss_entropy_beta=0.001, lmbda=0.95, train_epochs=10, exploration_noise=1.0, n_stack=1,
                 img_input=False, state_size=None, n_parallel_envs=None, save_base_dir=None, save_model_name=None,
                 save_each_n_iter=None, net_architecture=None, histogram_memory=False, tensorboard_dir=None):
        """
        Proximal Policy Optimization (PPO) for continuous action spaces with parallelized experience collection agent class.
        :param optimize_func: Objective function for sqp optimization.
        :param actor_lr: learning rate for training the actor NN of an Actor-Critic agent.
        :param critic_lr: learning rate for training the critic NN of an Actor-Critic agent.
        :param batch_size: batch size for training procedure.
        :param epsilon: exploration-exploitation rate during training. epsilon=1.0 -> Exploration, epsilon=0.0 -> Exploitation.
        :param epsilon_decay: exploration-exploitation rate decay factor.
        :param epsilon_min: min exploration-exploitation rate allowed during training.
        :param gamma: Discount factor for target value.
        :param n_step_return: Number of steps used for calculating the return.
        :param memory_size: Size of experiences memory.
        :param loss_clipping: Loss clipping factor for PPO.
        :param loss_critic_discount: Discount factor for critic loss of PPO.
        :param loss_entropy_beta: Discount factor for entropy loss of PPO.
        :param train_epochs: Train epoch for each training iteration.
        :param exploration_noise:
        :param n_stack: Number of time steps stacked on the state (observation stacked).
        :param img_input: Flag for using a images as states.
        :param state_size: State size. Needed if the original state size is modified by any preprocessing.
        :param n_parallel_envs: Number of parallel environments when using A3C or PPO.
        :param net_architecture: Net architecture.
        :param save_base_dir: Base directory for saving the agent network weights.
        :param save_model_name: Name of the model when saving the agent network weights.
        :param save_each_n_iter: Number of iterations between each saving.
        """
        super().__init__(actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size, epsilon=epsilon,
                         epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, gamma=gamma,
                         n_step_return=n_step_return, memory_size=memory_size, loss_clipping=loss_clipping,
                         loss_critic_discount=loss_critic_discount, loss_entropy_beta=loss_entropy_beta, lmbda=lmbda,
                         train_epochs=train_epochs, exploration_noise=exploration_noise, n_stack=n_stack,
                         img_input=img_input, state_size=state_size, n_parallel_envs=n_parallel_envs,
                         save_base_dir=save_base_dir, save_model_name=save_model_name,
                         save_each_n_iter=save_each_n_iter, net_architecture=net_architecture, histogram_memory=histogram_memory,
                         tensorboard_dir=tensorboard_dir)
        if self.n_parallel_envs is None:
            self.n_parallel_envs = multiprocessing.cpu_count()
        self.agent_name = agent_globals.names["ppooptcicles_continuous_parallel_v2"]
        self.opt_experiences_memory = deque()
        self.opt = "Nelder-Mead"  # Optimizer from ScyPy optimize.minimize()
        self.opt_maxiter = opt_maxiter

    def build_agent(self, state_size, n_actions, stack, action_bound=None):
        super().build_agent(state_size, n_actions, stack=stack)
        self.action_bound = action_bound
        self.loss_selected = self.proximal_policy_optimization_loss_continuous
        self.actor, self.critic = self._build_model(self.net_architecture, last_activation='linear')
        self.dummy_action, self.dummy_value = self.dummies_parallel(self.n_parallel_envs)
        self.remember = self.remember_parallel_opt

    def act_train(self, obs):
        obs = self._format_obs_act_parall(obs)
        p = self.actor.predict([obs, self.dummy_value, self.dummy_action, self.dummy_value, self.dummy_value])
        if np.random.rand() <= self.epsilon:
            action = action_matrix = p + np.random.normal(loc=0, scale=self.exploration_noise*self.epsilon, size=p.shape)
        else:
            action = action_matrix = p
        value = self.critic.predict(obs)

        return action, action_matrix, p, value

    def act(self, obs):
        obs = self._format_obs_act(obs)

        p = self.actor.predict([obs, self.dummy_value, self.dummy_action, self.dummy_value, self.dummy_value])
        action = p[0]
        return action

    def remember_parallel_opt(self, obs, action, pred_act, rewards, values, mask, opt_obs, opt_act, opt_act_prob,
                              opt_reward, opt_values, opt_mask):
        """
        Store a memory in a list of memories
        :param obs: Current Observation (State)
        :param action: Action selected with noise
        :param pred_act: Action predicted
        :param reward: Reward
        :param next_obs: Next Observation (Next State)
        :param done: If the episode is finished
        :return:
        """

        if self.img_input:
            obs = np.transpose(obs, axes=(1, 0, 2, 3, 4))
        elif self.stack:
            obs = np.transpose(obs, axes=(1, 0, 2, 3))
        else:
            obs = np.transpose(obs, axes=(1, 0, 2))

        action = np.transpose(action, axes=(1, 0, 2))
        pred_act = np.transpose(pred_act, axes=(1, 0, 2))
        rewards = np.transpose(rewards, axes=(1, 0))
        values = np.transpose(values, axes=(1, 0, 2))
        mask = np.transpose(mask, axes=(1, 0))

        o = obs[0]
        a = action[0]
        p_a = pred_act[0]
        r = rewards[0]
        v = values[0]
        m = mask[0]

        for i in range(1, self.n_parallel_envs):
            o = np.concatenate((o, obs[i]), axis=0)
            a = np.concatenate((a, action[i]), axis=0)
            p_a = np.concatenate((p_a, pred_act[i]), axis=0)
            r = np.concatenate((r, rewards[i]), axis=0)
            v = np.concatenate((v, values[i]), axis=0)
            m = np.concatenate((m, mask[i]), axis=0)

        for i in range(len(opt_obs)):
            o = np.concatenate((o, opt_obs[i]), axis=0)
            a = np.concatenate((a, opt_act[i]), axis=0)
            p_a = np.concatenate((p_a, opt_act_prob[i]), axis=0)
            r = np.concatenate((r, opt_reward[i]), axis=0)
            v = np.concatenate((v, opt_values[i]), axis=0)
            m = np.concatenate((m, opt_mask[i]), axis=0)

        v = np.concatenate((v, [v[-1]]), axis=0)
        returns, advantages = self.get_advantages(v, m, r)
        advantages = np.array(advantages)
        returns = np.array(returns)

        index = range(len(o))
        self.memory = [o[index], a[index], p_a[index], returns[index], r[index], v[index],
                       m[index], advantages[index]]
