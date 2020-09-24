from IRL_Problem.base.irl_problem_super import IRLProblemSuper
# from pympler import muppy, summary
# from memory_leaks import *
# from IRL.utils.parse_utils import *
# from src.IRL.Expert_Agent.expert import Expert
# from src.IRL.utils import callbacks
from IRL_Problem.base.networks import gail_discriminator
from RL_Agent.base.utils import agent_globals


class GAIL(IRLProblemSuper):
    """ Inverse Reinforcement Learning Problem.

    This class represent the src problem to solve.
    """

    def __init__(self, rl_problem, expert_traj, n_stack_disc=1, net_architecture=None, irl_params=None):
        """
        Attributes:
            environment:    Environment selected for this problem.
            rl_problem:     Problema de CAPORL relacionado.
            n_stack:        Int >= 1. If 1, there is no stacked input. Number of time related input stacked.
            img_input:      Bool. If True, input data is an image.
            state_size:     None, Int or Tuple. State dimensions. If None it will be calculated automatically. Int or
                            Tuple format will be useful when preprocessing change the input dimensions.
        """
        self._check_agent(rl_problem.agent)
        super().__init__(rl_problem, expert_traj, n_stack_disc, net_architecture=net_architecture, irl_params=irl_params)
        # TODO: check if agent is instance of ppo
        # self.discriminator = self._build_discriminator(net_architecture)

    def _build_discriminator(self, net_architecture):
        try:
            discrete_env = self.rl_problem.action_bound is None
        except:
            discrete_env = True

        n_stack = self.n_stack if self.n_stack_disc > 1 else 1
        return gail_discriminator.Discriminator("Discriminator", self.state_size, self.n_actions, n_stack=n_stack,
                                                img_input=self.img_input, expert_actions=self.action_memory,
                                                learning_rate=self.lr_disc, batch_size=self.batch_size_disc,
                                                epochs=self.epochs_disc, val_split=self.val_split_disc,
                                                discrete=discrete_env, net_architecture=net_architecture)

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
        self.rl_problem.solve(iterations, render=render, max_step_epi=None, render_after=None, skip_states=0,
                              discriminator=self.discriminator, expert_traj=self.expert_traj)

    def _check_agent(self, agent):
        valid_agent = agent.agent_name == agent_globals.names["ppo_discrete"] or \
                      agent.agent_name == agent_globals.names["ppo_continuous"] or \
                      agent.agent_name == agent_globals.names["ppo_discrete_parallel"] or \
                      agent.agent_name == agent_globals.names["ppo_continuous_parallel"]

        if not valid_agent:
            raise Exception('GAIL algorithm in this library only works with ppo rl agents but ' + str(agent.agent_name)
                            + ' rl agent was selected')
