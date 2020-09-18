from CAPORL.RL_Problem.rl_problem_super import *
from CAPORL.RL_Problem.PPO.ppo_problem_base import PPOProblemBase
import numpy as np
import cv2


class PPOProblem(PPOProblemBase):
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
                         model_params=model_params, saving_model_params=saving_model_params,
                         net_architecture=net_architecture, continuous=True)

    def _define_agent(self, agent, state_size, n_actions, stack, img_input, lr_actor, lr_critic, batch_size,
                      buffer_size, epsilon, epsilon_decay, epsilon_min, action_bound, net_architecture,
                      n_parallel_envs):
        return agent.Agent(state_size, n_actions, stack=stack, img_input=img_input, lr_actor=lr_actor,
                           lr_critic=lr_critic, action_bound=action_bound, batch_size=batch_size,
                           buffer_size=buffer_size, epsilon=epsilon, epsilon_decay=epsilon_decay,
                           epsilon_min=epsilon_min, net_architecture=net_architecture)
    def reduce_exploration_noise(self):
        """
        For continuous action spaces
        """
        self.agent.exploration_noise = np.random.rand() * 2 / 10