import numpy as np
from RL_Agent.base.PPO_base.ppo_agent_base import PPOSuper
from RL_Agent.base.utils import agent_globals
import multiprocessing
import psutil
import matplotlib.pyplot as plt

def create_agent():
    return agent_globals.names["ppo_continuous_parallel_hindsight"]

class Agent(PPOSuper):
    def __init__(self, actor_lr=1e-4, critic_lr=1e-3, batch_size=32, epsilon=1.0, epsilon_decay=1., epsilon_min=0.1,
                 gamma=0.95, n_step_return=10, memory_size=512, loss_clipping=0.2, loss_critic_discount=0.5,
                 loss_entropy_beta=0.001, lmbda=0.95, train_epochs=10, exploration_noise=1.0, n_stack=1,
                 img_input=False, state_size=None, n_parallel_envs=None, save_base_dir=None, save_model_name=None,
                 save_each_n_iter=None, net_architecture=None, histogram_memory=False, tensorboard_dir=None):
        """
        Proximal Policy Optimization (PPO) for continuous action spaces with parallelized experience collection agent class.
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
        self.agent_name = agent_globals.names["ppo_continuous_parallel"]

    def build_agent(self, state_size, n_actions, stack, action_bound=None):
        super().build_agent(state_size, n_actions, stack=stack)

        self.action_bound = action_bound
        self.loss_selected = self.proximal_policy_optimization_loss_continuous
        self.actor, self.critic = self._build_model(self.net_architecture, last_activation='linear')
        self.dummy_action, self.dummy_value = self.dummies_parallel(self.n_parallel_envs)
        self.remember = self.remember_parallel

    def act_train(self, obs):
        obs = self._format_obs_act_parall(obs)
        p = self.actor.predict([obs, self.dummy_value, self.dummy_action, self.dummy_value, self.dummy_value])
        action = action_matrix = p + np.random.normal(loc=0, scale=self.exploration_noise*self.epsilon, size=p.shape)
        value = self.critic.predict(obs)
        return action, action_matrix, p, value

    def act(self, obs):
        obs = self._format_obs_act(obs)

        p = self.actor.predict([obs, self.dummy_value, self.dummy_action, self.dummy_value, self.dummy_value])
        action = p[0]
        return action

    def load_memories(self, number_of_opt_experiences=0):
        """
        Load a batch of memories
        :return: Current Observation, Action, Reward, Next Observation, episode finished flag
        """
        obs = self.memory[0]
        action = self.memory[1]
        pred_act = self.memory[2]
        returns = self.memory[3]
        rewards = self.memory[4]
        values = self.memory[5]
        mask = self.memory[6]
        advantages = self.memory[7]

        rewards_for_tb = rewards[number_of_opt_experiences:]
        returns_for_tb = returns[number_of_opt_experiences:]
        values_for_tb = values[number_of_opt_experiences:]
        advantages_for_tb = advantages[number_of_opt_experiences:]
        action_for_tb = action[number_of_opt_experiences:]

        hind_obs = []
        hind_action = []
        hind_pred_act = []
        hind_returns = []
        hind_rewards = []
        hind_values = []
        hind_mask = []
        hind_advantages = []

        first_index_episode = 0
        for i in range(mask.shape[0]):
            if not mask[i]:
                goal = obs[i, :, 5:161]
                for j in range(i, first_index_episode, -1):
                    hind_obs.append(np.concatenate([obs[j, :, :161], goal], axis=-1))
                    hind_action.append(action[j])
                    hind_pred_act.append(pred_act[j])
                    hind_returns.append(returns[j])
                    if i == j:
                        hind_rewards.append(0.0)
                    else:
                        hind_rewards.append(rewards[j])
                    hind_values.append(values[j])
                    hind_mask.append(mask[j])
                    hind_advantages.append(advantages[j])
                first_index_episode = i + 1

        obs = np.concatenate([obs, hind_obs], axis=0)
        action = np.concatenate([action, hind_action], axis=0)
        pred_act = np.concatenate([pred_act, hind_pred_act], axis=0)
        returns = np.concatenate([returns, hind_returns], axis=0)
        rewards = np.concatenate([rewards, hind_rewards], axis=0)
        values = np.concatenate([values, hind_values], axis=0)
        mask = np.concatenate([mask, hind_mask], axis=0)
        advantages = np.concatenate([advantages, hind_advantages], axis=0)
        if self.histogram_memory:
            ###########################################################################################
            #           Selecting experiences by histogram
            ###########################################################################################
            # Auto histogram

            aux_rew = rewards.copy()
            aux_rew[aux_rew < 0] = -0.2
            max = 1.5
            min = -0.2
            n_bins = 30
            plt.figure(1)
            plt.clf()
            _ = plt.hist(aux_rew, range=(min, max), bins=n_bins)
            # plt.figure(2)
            # plt.clf()
            # _ = plt.hist(aux_rew, range=(0., max), bins='auto')
            # plt.show()


            hist, bin_edges = np.histogram(aux_rew, range=(min, max), bins=n_bins)
            # extract bin index for each array position
            gfg = np.digitize(aux_rew, bin_edges) - 1
            gfg[gfg == -1] = bin_edges.shape[0]
            # Reorder the index by bins
            bin_index = np.array([np.arange(gfg.shape[0]), gfg], dtype=np.int32)
            bins = [[] for i in range(bin_edges.shape[0]+1)]
            for index in range(gfg.shape[0]):
                # gives a single float value
                psutil.cpu_percent()
                # gives an object with many fields
                psutil.virtual_memory()
                # you can convert that object to a dictionary
                dict(psutil.virtual_memory()._asdict())
                # you can have the percentage of used RAM
                ram_usage = psutil.virtual_memory().percent
                if ram_usage > 75.:
                    print('ram usage: ', ram_usage)

                bins[bin_index[1][index]].append(bin_index[0][index])

            _obs = []
            _action = []
            _pred_act = []
            _returns = []
            _rewards = []
            _values = []
            _mask = []
            _advantages = []

            # Count empty bins
            empty_bins_count = 0
            for i in range(len(bins)):
                # gives a single float value
                psutil.cpu_percent()
                # gives an object with many fields
                psutil.virtual_memory()
                # you can convert that object to a dictionary
                dict(psutil.virtual_memory()._asdict())
                # you can have the percentage of used RAM
                ram_usage = psutil.virtual_memory().percent
                if ram_usage > 75.:
                    print('ram usage: ', ram_usage)

                if len(bins[i]) == 0:
                    empty_bins_count += 1
            n_valid_bins = bin_edges.shape[0] + 1 - empty_bins_count

            # Sampling uniformly from bins
            for i in range(int(rewards.shape[0] / n_valid_bins)):
                for j in range(bin_edges.shape[0]):
                    # gives a single float value
                    psutil.cpu_percent()
                    # gives an object with many fields
                    psutil.virtual_memory()
                    # you can convert that object to a dictionary
                    dict(psutil.virtual_memory()._asdict())
                    # you can have the percentage of used RAM
                    ram_usage = psutil.virtual_memory().percent
                    if ram_usage > 75.:
                        print('ram usage: ', ram_usage)
                    if len(bins[j]) > 0:
                        index = np.random.choice(bins[j])
                        _obs.append(obs[index])
                        _action.append(action[index])
                        _pred_act.append(pred_act[index])
                        _returns.append(returns[index])
                        _rewards.append(rewards[index])
                        _values.append(values[index])
                        _mask.append(mask[index])
                        _advantages.append(advantages[index])

            obs = np.array(_obs)
            action = np.array(_action)
            pred_act = np.array(_pred_act)
            returns = np.array(_returns)
            rewards = np.array(_rewards)
            values = np.array(_values)
            mask = np.array(_mask)
            advantages = np.array(_advantages)
            ###########################################################################################
            #           done Selecting experiences by histogram
            ###########################################################################################

            # plt.figure(2)
            # plt.clf()
            # aux_rew = rewards.copy()
            # aux_rew[aux_rew < 0] = -0.2
            # _ = plt.hist(aux_rew, range=(min, max), bins=n_bins)
            plt.draw()
            plt.pause(10e-50)
        return obs, action, pred_act, returns, rewards, values, mask, advantages, rewards_for_tb, action_for_tb, returns_for_tb, advantages_for_tb, values_for_tb

