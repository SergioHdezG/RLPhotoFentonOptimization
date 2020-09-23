from collections import deque

# from pympler import muppy, summary
# from memory_leaks import *
# from IRL.utils.parse_utils import *
# from src.IRL.Expert_Agent.expert import Expert
# from src.IRL.utils import callbacks
from IRL_Problem.base.utils.callbacks import Callbacks


class IRLProblemSuper(object):
    """ Inverse Reinforcement Learning Problem.

    This class represent the src problem to solve.
    """

    def __init__(self, rl_problem, expert_traj, n_stack_disc=1):
        """
        Attributes:
            environment:    Environment selected for this problem.
            rl_problem:     Problema de CAPORL relacionado.
            n_stack:        Int >= 1. If 1, there is no stacked input. Number of time related input stacked.
            img_input:      Bool. If True, input data is an image.
            state_size:     None, Int or Tuple. State dimensions. If None it will be calculated automatically. Int or
                            Tuple format will be useful when preprocessing change the input dimensions.
        """
        self.n_stack = rl_problem.n_stack
        self.img_input = rl_problem.img_input

        self.state_size = rl_problem.state_size
        self.n_actions = rl_problem.n_actions

        if self.n_stack < 2 and n_stack_disc > 1:
            raise Exception("Is not allowed to use stacked input for the discriminator if agent do not use stacked "
                            "inputs. Discriminator: n_stack_disc = "+str(n_stack_disc)+", Agent: n_stack = "
                            + str(self.n_stack))
        elif self.n_stack > 1 and self.n_stack != n_stack_disc and n_stack_disc != 1:
            raise Exception("Is not allowed to use stacked input for both discriminator and agent if number of stacked "
                            "inputs differ. It is allowed to use the same stacking number for both or use stacked input"
                            "for agent but don't use it for discriminator. Discriminator: n_stack_disc = "
                            + str(n_stack_disc) + ", Agent: n_stack = " + str(self.n_stack))
        else:
            self.n_stack_disc = n_stack_disc

        # Reinforcement learning problem/agent
        self.rl_problem = rl_problem
        # self.agent_traj = None
        self.agent_traj = deque(maxlen=20000)
        self.expert_traj = expert_traj

        # If expert trajectories includes the action took for the expert: True, if only include the observations: False
        self.action_memory = len(self.expert_traj[0]) > 1

        # Total number of steps processed
        self.global_steps = 0

        self.max_rew_mean = -100000  # Store the maximum value for reward mean

        self.discriminator = self._build_discriminator()

    def _build_discriminator(self):
        pass

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
        if False:
            for iter in range(iterations):
                    n_agent_iter = 10
                    # self.agent_traj = self.agent_play(n_agent_iter, render=render)

                    for element in self.agent_play(n_agent_iter, render=render):
                        self.agent_traj.append(element)

                    self.discriminator.train(self.expert_traj, self.agent_traj)

                    self.rl_problem.solve(100, render=render, max_step_epi=None, render_after=1000, skip_states=0, discriminator=self.discriminator)
        else:
            self.rl_problem.solve(iterations, render=render, max_step_epi=None, render_after=None, skip_states=0, discriminator=self.discriminator,
                                  expert_traj=self.expert_traj)


    def load_expert_data(self):
        pass

    def agent_play(self, n_iter, render=False):
        # Callback to save in memory agent trajectories
        callback = Callbacks()
        self.rl_problem.test(render=render, n_iter=n_iter, callback=callback.remember_callback)
        return callback.get_memory(self.action_memory, self.discriminator.n_stack)


    def _preprocess(self, obs):
        return obs

    def _clip_norm_reward(self, rew):
        return rew

