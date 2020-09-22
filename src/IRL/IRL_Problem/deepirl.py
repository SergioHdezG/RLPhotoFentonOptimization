from collections import deque
from src.IRL.IRL_Problem.base.irl_problem_super import IRLProblemSuper
import numpy as np
import datetime as dt
# from pympler import muppy, summary
# from memory_leaks import *
# from IRL.utils.parse_utils import *
# from src.IRL.Expert_Agent.expert import Expert
# from src.IRL.utils import callbacks
from src.IRL.utils.callbacks import Callbacks
from src.IRL.networks import vanilla_deep_irl, gail, gail_v2, gail_discriminator
import gym

class DeepIRL(IRLProblemSuper):
    """ Inverse Reinforcement Learning Problem.

    This class represent the src problem to solve.
    """

    def __init__(self, rl_problem, expert_traj, n_stack_disc=False):
        """
        Attributes:
            environment:    Environment selected for this problem.
            rl_problem:     Problema de CAPORL relacionado.
            n_stack:        Int >= 1. If 1, there is no stacked input. Number of time related input stacked.
            img_input:      Bool. If True, input data is an image.
            state_size:     None, Int or Tuple. State dimensions. If None it will be calculated automatically. Int or
                            Tuple format will be useful when preprocessing change the input dimensions.
        """
        super().__init__(rl_problem, expert_traj, n_stack_disc)
        self.discriminator = self._build_discriminator()

    def _build_discriminator(self):
        try:
            discrete_env = self.rl_problem.action_bound is None
        except:
            discrete_env = True

        n_stack = self.n_stack if self.n_stack_disc > 1 else 1
        return vanilla_deep_irl.Discriminator("Discriminator", self.state_size, self.n_actions, n_stack=n_stack,
                                                img_input=self.img_input, expert_actions=self.action_memory, learning_rate=1e-5,
                                                discrete=discrete_env)

    def solve(self, iterations, render=True, render_after=None, max_step_epi=None, skip_states=1,
              verbose=1):
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
        for iter in range(iterations):
                n_agent_iter = 10
                # self.agent_traj = self.agent_play(n_agent_iter, render=render)

                for element in self.agent_play(n_agent_iter, render=render):
                    self.agent_traj.append(element)

                self.discriminator.train(self.expert_traj, self.agent_traj)

                self.rl_problem.solve(100, render=render, max_step_epi=None, render_after=1000, skip_states=0, discriminator=self.discriminator)


