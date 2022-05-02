from RL_Problem.base.rl_problem_base import *
from RL_Problem.base.PPO.multiprocessing_env import SubprocVecEnv
from RL_Problem.base.PPO.ppo_problem_base import PPOProblemBase
import multiprocessing
import numpy as np
import copy


def create_agent():
    return "PPO_continuous"

class PPOProblemParallelBase(PPOProblemBase):
    """
    Proximal Policy Optimization.
    """
    def __init__(self, environment, agent, continuous=False):
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

        super().__init__(environment, agent, continuous=continuous)

        # Copiamos el entorno creado en rl_problem_super para que lo almacene env_test
        self.env_test = self.env
        self.n_parallel_envs = self.agent.n_parallel_envs
        # Creamos n copias del entorno
        env = [self.make_env() for i in range(self.n_parallel_envs)]
        self.env = SubprocVecEnv(env)
        self.run_test = True

    def make_env(self):
        # returns a function which creates a single environment
        def _thunk():
            # Inicializar el entorno
            try:
                env = copy.deepcopy(self.env)
            except:
                env = self.env.copy()
            return env
        return _thunk

    def _define_agent(self, n_actions, state_size, stack, action_bound=None):
        pass

    def collect_batch(self, render, render_after, max_step_epi, skip_states, verbose, discriminator=None, expert_traj=None):
        self.agent.best_params_fotocaos_list = []

        self.obs_batch = []
        self.actions_batch = []
        self.actions_probs_batch = []
        self.rewards_batch = []
        self.values_batch = []
        self.masks_batch = []
        # Stacking inputs
        if self.n_stack is not None and self.n_stack > 1:
            obs_queue = [deque(maxlen=self.n_stack) for i in range(self.n_parallel_envs)]
            obs_next_queue = [deque(maxlen=self.n_stack) for i in range(self.n_parallel_envs)]
        else:
            obs_queue = None
            obs_next_queue = None

        if self.run_test:
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

            # Agent act in the environment
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

    def store_episode_experience(self, action, done, next_obs, obs, obs_next_queue, obs_queue, reward, skip_states, epochs,
                                predicted_action, action_matrix, value):

        done, next_obs, reward, epochs = self.frame_skipping(action, done, next_obs, reward, skip_states, epochs)

        # Con este método de paralelización no se puede aplicar frame-skipping
        mask = np.logical_not(done)

        if self.n_stack is not None and self.n_stack > 1:
            for o, next_queue in zip(next_obs, obs_next_queue):
                next_queue.append(o)

            if self.img_input:
                obs_queue_stack = np.array([np.dstack(o) for o in obs_queue])
            else:
                # obs_next_stack = np.array(obs_next_queue).reshape(self.n_asyn_envs, self.state_size, self.n_stack)
                obs_queue_stack = np.array(obs_queue)
            self.obs_batch.append(obs_queue_stack)
        else:
            self.obs_batch.append(obs)

        self.actions_batch.append(action_matrix)
        self.actions_probs_batch.append(predicted_action)
        self.rewards_batch.append(reward)
        self.values_batch.append(value)
        self.masks_batch.append(mask)

        return next_obs, obs_next_queue, reward, done, epochs, mask

    def frame_skipping(self, action, done, next_obs, reward, skip_states, epochs):
        # isdone = False
        # for d in done:
        #     if d:
        #         isdone = True
        #
        # if isdone:
        #     print('is done')
        if skip_states > 1 and not done.any():
            for i in range(skip_states - 2):
                next_obs_aux1, reward_aux, done_aux, _ = self.env.step(action)
                epochs += 1
                reward += reward_aux
                if done_aux.any():
                    next_obs_aux2 = next_obs_aux1
                    done = done_aux
                    break

            if not done.any():
                next_obs_aux2, reward_aux, done_aux, _ = self.env.step(action)
                epochs += 1
                reward += reward_aux
                done = done_aux

            # if self.img_input:
            #     next_obs_aux2 = self.preprocess(next_obs_aux2)
            #     if skip_states > 2:
            #         next_obs_aux1 = self.preprocess(next_obs_aux1)
            #         next_obs = np.maximum(next_obs_aux2, next_obs_aux1)
            #     else:
            #         next_obs = self.preprocess(next_obs)
            #         next_obs = np.maximum(next_obs_aux2, next_obs)
            # else:
            next_obs = np.array([self.preprocess(o) for o in next_obs_aux2])
        else:
            next_obs = np.array([self.preprocess(o) for o in next_obs])
        return done, next_obs, reward, epochs

    def test(self, n_iter=10, render=True, callback=None):
        # Reasignamos el entorno de test al entorno principal para poder usar el método test del padre
        aux_env = self.env
        self.env = self.env_test
        super().test(n_iter=n_iter, render=render, callback=callback)
        self.env = aux_env