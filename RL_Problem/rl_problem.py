from RL_Problem.base.PPO import ppo_problem_continuous_parallel, ppo_problem_discrete, \
    ppo_opt_cicles_problem_continuous_parallel_v2
from RL_Agent.base.utils import agent_globals


def Problem(environment, agent):
    """ Class for selecting an algorithm to use
    :param environment: Environment selected.
    :param agent:       String. ID of the agent to use.
    :param n_stack:     Int >= 1. If 1, there is no stacked input. Number of time related input stacked.
    :param img_input:   Bool. If True, input data is an image.
    :param state_size:  None, Int or Tuple. State dimensions. If None it will be calculated automatically. Int or
                        Tuple format will be useful when preprocessing change the input dimensions.
    :return:
    """
    if agent.agent_name == agent_globals.names["ppo_continuous_parallel"]:
        problem = ppo_problem_continuous_parallel.PPOProblem(environment, agent)
    elif agent.agent_name == agent_globals.names["ppooptcicles_continuous_parallel_v2"]:
        problem = ppo_opt_cicles_problem_continuous_parallel_v2.PPOProblem(environment, agent)

    return problem
