from CAPORL.RL_Problem.rl_problem_super import *
import numpy as np

def create_agent():
    return "PPO_continuous"

class PPOProblem(RLProblemSuper):
    """
    Proximal Policy Optimization.
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
        super().__init__(environment, agent, n_stack=n_stack, img_input=img_input, state_size=state_size,
                         saving_model_params=saving_model_params, net_architecture=net_architecture)
        self.environment = environment

        if model_params is not None:
            batch_size, epsilon, epsilon_min, epsilon_decay, learning_rate, n_step_rew = \
                parse_model_params(model_params)

        self.action_bound = [self.env.action_space.low, self.env.action_space.high]  # action bounds

        # self.critic = self.build_critic()

        # self.actor = self.build_actor_continuous()

        # self.env = gym.make(ENV)
        # print(self.env.action_space, 'action_space', self.env.observation_space, 'observation_space')
        self.episode = 0
        self.val = False
        self.reward = []
        self.reward_over_time = []

        self.gradient_steps = 0

        # if "A2C_continuous" in agent.create_agent():
        self.action_bound = [self.env.action_space.low, self.env.action_space.high]  # action bounds

        self.batch_size = batch_size
        self.buffer_size = 2048
        self.learning_rate = learning_rate
        # List of 100 last rewards
        self.rew_mean_list = deque(maxlen=100)
        self.global_steps = 0

        self.agent = self._build_agent(agent, model_params, net_architecture)



    def _build_agent(self, agent, model_params, net_architecture):
        if self.img_input:
            stack = self.n_stack is not None and self.n_stack > 1
            state_size = (*self.state_size[:2], self.state_size[-1] * self.n_stack)

        elif self.n_stack is not None and self.n_stack > 1:
            stack = True
            state_size = (self.state_size, self.n_stack)
        else:
            stack = False
            state_size = self.state_size

        return agent.Agent(state_size, self.n_actions, stack=stack,
                                       img_input=self.img_input, lr_actor=self.learning_rate, lr_critic=self.learning_rate,
                                       action_bound=self.action_bound, batch_size=self.batch_size,
                                       buffer_size=self.buffer_size, net_architecture=net_architecture)

    def solve(self, episodes, render=True, render_after=None, max_step_epi=None, skip_states=1, verbose=1):
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
        self.rew_mean_list = deque(maxlen=100)

        while self.episode < episodes:
            self.collect_batch(render, render_after, max_step_epi, skip_states, verbose)
            actor_loss, critic_loss = self.agent.replay()

            print('Actor loss', actor_loss.history['loss'][-1], self.gradient_steps)
            print('Critic loss', critic_loss.history['loss'][-1], self.gradient_steps)

            self.gradient_steps += 1

    def collect_batch(self, render, render_after, max_step_epi, skip_states, verbose):
        batch = [[], [], [], []]

        # Stacking inputs
        if self.n_stack is not None and self.n_stack > 1:
            obs_queue = deque(maxlen=self.n_stack)
            obs_next_queue = deque(maxlen=self.n_stack)
        else:
            obs_queue = None
            obs_next_queue = None

        while len(batch[0]) < self.buffer_size:
            tmp_batch = [[], [], []]

            if self.episode % 100 == 0:
                self.test(n_iter=1, render=True)

            obs = self.env.reset()
            episodic_reward = 0
            epochs = 0
            done = False
            self.reward = []

            obs = self.preprocess(obs)

            # Stacking inputs
            if self.n_stack is not None and self.n_stack > 1:
                for i in range(self.n_stack):
                    obs_queue.append(obs)
                    obs_next_queue.append(obs)

            while not done:  # and len(batch[0])+len(tmp_batch[0]) < self.buffer_size:
                if render or ((render_after is not None) and self.episode > render_after):
                    self.env.render()

                # Select an action
                action, action_matrix, predicted_action = self.act(obs, obs_queue)

                # Agent act in the environment
                next_obs, reward, done, info = self.env.step(action)

                # Store the experience in episode memory
                next_obs, obs_next_queue, reward, done, epochs, tmp_batch = self.store_experience(action, done,
                                                                                                  next_obs, obs,
                                                                                                  obs_next_queue,
                                                                                                  obs_queue, reward,
                                                                                                  skip_states, epochs,
                                                                                                  tmp_batch,
                                                                                                  predicted_action,
                                                                                                  action_matrix)
                # self.reward.append(reward)
                # tmp_batch[0].append(obs)
                # tmp_batch[1].append(action_matrix)
                # tmp_batch[2].append(predicted_action)

                # copy next_obs to obs
                obs, obs_queue = self.copy_next_obs(next_obs, obs, obs_next_queue, obs_queue)

                episodic_reward += reward
                epochs += 1
                self.global_steps += 1

            # print('Episode reward', np.array(self.reward).sum(), self.episode)

            reward_return = self.agent.compute_reward_return(self.reward)

            for i in range(len(tmp_batch[0])):
                obs, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
                r = reward_return[i]
                batch[0].append(obs)
                batch[1].append(action)
                batch[2].append(pred)
                batch[3].append(r)

            self.episode += 1

            # Add reward to the list
            self.rew_mean_list.append(episodic_reward)

            print('batch size: ', len(batch[0]))
            # Print log on scream
            self._feedback_print(self.episode, episodic_reward, epochs, verbose, self.rew_mean_list)

        obs_batch, action_batch, pred_batch, reward_batch = batch[0], batch[1], batch[2], batch[3]
        self.agent.remember(obs_batch, action_batch, pred_batch, reward_batch)

    def store_experience(self, action, done, next_obs, obs, obs_next_queue, obs_queue, reward, skip_states, epochs,
                         tmp_batch, predicted_action, action_matrix):

        done, next_obs, reward, epochs = self.frame_skipping(action, done, next_obs, reward, skip_states, epochs)

        if self.n_stack is not None and self.n_stack > 1:
            obs_next_queue.append(next_obs)

            if self.img_input:
                obs_next_stack = np.dstack(obs_next_queue)
            else:
                obs_next_stack = np.array(obs_next_queue).reshape(self.state_size, self.n_stack)
            tmp_batch[0].append(obs_next_stack)
        else:
            tmp_batch[0].append(obs)

        self.reward.append(reward)
        tmp_batch[1].append(action_matrix)
        tmp_batch[2].append(predicted_action)

        return next_obs, obs_next_queue, reward, done, epochs, tmp_batch