from RL_Problem.base.PPO.ppo_problem_parallel_base import PPOProblemParallelBase
from collections import deque
import numpy as np
from src.environments import perox_sqp_globals

def create_agent():
    return "PPO_continuous"

class PPOProblem(PPOProblemParallelBase):
    """
    Proximal Policy Optimization.
    """
    def __init__(self, environment, agent):
        """
        This problem uses the PPOSQP agent from RL_Agent.pposqp_agent_continuous_parallel.py agent and environment from
        src.perox_model_small_sqp_hybrid.py which has an special signature for the step method receiving: action,
        action_matrix, predicted_action, value and the agent critic network (self.agent.critic) and returning the next
        values: next_obs, reward, done, info, action, action_matrix, predicted_action, state_value.
        Attributes:
                environment:    Environment selected for this problem
                agent:          Agent to solve the problem: DDPG.
                n_stack:        Int >= 1. If 1, there is no stacked input. Number of time related input stacked.
                img_input:      Bool. If True, input data is an image.
                state_size:     None, Int or Tuple. State dimensions. If None it will be calculated automatically. Int
                                or Tuple format will be useful when preprocessing change the input dimensions.
                model_params:   Dictionary of params like learning rate, batch size, epsilon values, n step returns...
        """
        super().__init__(environment, agent, continuous=True)

        # Memorias de experiencias del optimizador
        self.opt_obs = []
        self.opt_reward = []
        self.opt_done = []
        self.opt_params = []

    def _define_agent(self, n_actions, state_size, stack, action_bound):
        self.agent.build_agent(state_size, n_actions, stack=stack, action_bound=action_bound)
        perox_sqp_globals.state_size = self.state_size

    def collect_batch(self, render, render_after, max_step_epi, skip_states, verbose, discriminator=None, expert_traj=None):
        self.agent.best_params_fotocaos_list = []
        obs, reward, done, info, params, error = self.env.record_opt_experiences(self.agent.opt, self.agent.opt_maxiter)

        # if self.img_input:
        #     obs = np.transpose(obs, axes=(1, 0, 2, 3))
        # else:
        #     obs = np.transpose(obs, axes=(1, 0, 2))
        #
        # params = np.transpose(params, axes=(1, 0, 2))
        # reward = np.transpose(reward, axes=(1, 0))
        #
        # done = np.transpose(done, axes=(1, 0))

        # # Stacking inputs
        # if self.n_stack is not None and self.n_stack > 1:
        #     obs_queue = [deque(maxlen=self.n_stack) for i in range(self.n_parallel_envs)]
        #     obs_next_queue = [deque(maxlen=self.n_stack) for i in range(self.n_parallel_envs)]
        # else:
        #     obs_queue = None
        #     obs_next_queue = None

        obs_aux = []
        for i in range(len(obs)):
            obs_aux.append(np.array([self.preprocess(o) for o in obs[i]]))
        obs = obs_aux


        self.opt_obs.append(obs)
        self.opt_reward.append(reward)
        self.opt_done.append(done)
        self.opt_params.append(params)

        self.number_of_optimizer_experiences = np.array(obs).shape[0] * np.array(self.opt_obs).shape[1] * np.array(self.opt_obs).shape[2]
        self.obs_batch = []
        self.actions_batch = []
        self.actions_probs_batch = []
        self.rewards_batch = []
        self.values_batch = []
        self.masks_batch = []

        for (obs, reward, done, params) in zip(self.opt_obs, self.opt_reward, self.opt_done, self.opt_params):
            # Stacking inputs
            if self.n_stack is not None and self.n_stack > 1:
                for i in range(self.n_stack):
                    zero_obs = np.zeros(obs[0][0].shape)
                    for queue, next_queue in zip(obs_queue, obs_next_queue):
                        [queue.append(zero_obs) for i in range(self.n_stack)]
                        [next_queue.append(zero_obs) for i in range(self.n_stack)]
                    for o, queue, next_queue in zip(obs[0], obs_queue, obs_next_queue):
                        queue.append(o)
                        next_queue.append(o)


            for i in range(1, len(obs)):
                values = self.agent.critic.predict(np.array(obs_queue))
                _, obs_next_queue, _, _, _, _ = self.store_episode_experience(params[i],
                                                                                done[i],
                                                                                obs[i],
                                                                                obs[i-1],
                                                                                obs_next_queue,
                                                                                obs_queue,
                                                                                reward[i],
                                                                                0,
                                                                                0,
                                                                                params[i],
                                                                                params[i],
                                                                                values)
                obs_queue = obs_next_queue.copy()

        ###############################################################################
        #   Mantener o no las experiencias del optimizador durante varias iteraciones
        self.opt_obs = []
        self.opt_reward = []
        self.opt_done = []
        self.opt_params = []
        #################################################################################
        # perox_sqp_globals.critic_nn = self.agent.critic

        # Stacking inputs
        if self.n_stack is not None and self.n_stack > 1:
            obs_queue = [deque(maxlen=self.n_stack) for i in range(self.n_parallel_envs)]
            obs_next_queue = [deque(maxlen=self.n_stack) for i in range(self.n_parallel_envs)]
        else:
            obs_queue = None
            obs_next_queue = None

        if self.run_test:
            # self.test(n_iter=5, render=True)
            self.run_test = False

        obs = self.env.reset()
        episodic_reward = 0
        epochs = 0
        done = False
        self.reward = []

        obs = np.array([self.preprocess(o) for o in obs])

        # Stacking inputs
        if self.n_stack is not None and self.n_stack > 1:
            for i in range(self.n_stack):
                zero_obs = np.zeros(obs[0].shape)
                for o, queue, next_queue in zip(obs, obs_queue, obs_next_queue):
                    [queue.append(zero_obs) for i in range(self.n_stack)]
                    [next_queue.append(zero_obs) for i in range(self.n_stack)]
                for o, queue, next_queue in zip(obs, obs_queue, obs_next_queue):
                    queue.append(o)
                    next_queue.append(o)

        for iter in range(self.memory_size):
            if render or ((render_after is not None) and self.episode > render_after):
                self.env.render()

            # Select an action
            action, action_matrix, predicted_action, value = self.act_train(obs, obs_queue)

            next_obs, reward, done, info = self.env.step(action)

            for item in info:
                if item is not None:
                    self.agent.best_params_fotocaos_list.append(item)

            if discriminator is not None:
                if discriminator.stack:
                    reward = discriminator.get_reward(obs_queue, action, parallel=True)
                else:
                    reward = discriminator.get_reward(obs, action, parallel=True)

            # Store the experience in episode memory
            next_obs, obs_next_queue, reward, done, epochs, mask = self.store_episode_experience(action,
                                                                                                done,
                                                                                                next_obs,
                                                                                                obs,
                                                                                                obs_next_queue,
                                                                                                obs_queue,
                                                                                                reward,
                                                                                                skip_states,
                                                                                                epochs,
                                                                                                predicted_action,
                                                                                                action_matrix,
                                                                                                value)

            # copy next_obs to obs
            obs, obs_queue = self.copy_next_obs(next_obs, obs, obs_next_queue, obs_queue)

            episodic_reward += reward
            epochs += 1
            self.global_steps += 1

        # Add reward to the list
        self.rew_mean_list.append(episodic_reward)
        rew_mean = [np.mean(self.rew_mean_list[i]) for i in range(len(self.rew_mean_list))]
        # Print log on scream
        for i_print in range(self.n_parallel_envs):
            # rew_mean_list = [rew[i_print] for rew in self.rew_mean_list]
            self._feedback_print(self.episode, episodic_reward[i_print], epochs, verbose, rew_mean)
            self.episode += 1
            if self.episode % 99 == 0:
                self.run_test = True

        if discriminator is not None and expert_traj is not None:
            if self.img_input:
                obs = np.transpose(self.obs_batch, axes=(1, 0, 2, 3, 4))
            elif self.n_stack > 1:
                obs = np.transpose(self.obs_batch, axes=(1, 0, 2, 3))
            else:
                obs = np.transpose(self.obs_batch, axes=(1, 0, 2))
            action = np.transpose(self.actions_batch, axes=(1, 0, 2))

            o = obs[0]
            a = action[0]
            for i in range(1, self.n_parallel_envs):
                o = np.concatenate((o, obs[i]), axis=0)
                a = np.concatenate((a, action[i]), axis=0)

            obs = o
            action = a

            if discriminator.stack:
                agent_traj = [[np.array(o), np.array(a)] for o, a in zip(obs, action)]
            else:
                # agent_traj = [[np.array(o[-1, :]), np.array(a)] for o, a in zip(obs, action)]
                agent_traj = [[np.array(o), np.array(a)] for o, a in zip(obs, action)]

            discriminator.train(expert_traj, agent_traj)

            if discriminator.stack:
                self.rewards_batch = [discriminator.get_reward(o, a, parallel=True) for o, a in zip(self.obs_batch, self.actions_batch)]
            else:
                self.rewards_batch = [discriminator.get_reward(o, a, parallel=True) for o, a in zip(self.obs_batch, self.actions_batch)]


        self.agent.remember(self.obs_batch, self.actions_batch, self.actions_probs_batch, self.rewards_batch,
                            self.values_batch, self.masks_batch)
    #
    # def test(self, n_iter=10, render=True, callback=None):
    #     # Reasignamos el entorno de test al entorno principal para poder usar el método test del padre
    #     aux_env = self.env
    #     self.env = self.env_test
    #     super().test(n_iter=n_iter, render=render, callback=callback)
    #     self.env = aux_env

    def test(self, n_iter=10, render=True, callback=None):
        """ Test a trained agent on an environment
        :param n_iter: int. number of test iterations
        :param name_loaded: string. Name of file of model to load. If empty, no model will be loaded.
        :param render: bool. Render or not
        :return:
        """
        # Reasignamos el entorno de test al entorno principal para poder usar el método test del padre
        aux_env = self.env
        self.env = self.env_test
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
                action = self.act(obs, obs_queue)
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

        self.env = aux_env
        # print('Mean Reward ', epi_rew_mean / n_iter)
        self.env.close()