from RL_Problem.ActorCritic import a3c_problem, a2c_problem, ddpg_problem
from RL_Problem.ValueBased import dqn_problem
from RL_Problem.PolicyBased import dpg_problem
from RL_Problem.PPO import ppo_problem_continuous, ppo_problem_continuous_parallel, ppo_problem_discrete, ppo_problem_discrete_parallel
from RL_Agent.base.utils import agent_globals


def Problem(environment, agent, model_params, saving_model_params=None, net_architecture=None, n_stack=1, img_input=False, state_size=None):
    """ Class for selecting an algorithm to use
    :param environment: Environment selected.
    :param agent:       String. ID of the agent to use.
    :param n_stack:     Int >= 1. If 1, there is no stacked input. Number of time related input stacked.
    :param img_input:   Bool. If True, input data is an image.
    :param state_size:  None, Int or Tuple. State dimensions. If None it will be calculated automatically. Int or
                        Tuple format will be useful when preprocessing change the input dimensions.
    :return:
    """
    if agent.agent_name == agent_globals.names["dqn"]:
        problem = dqn_problem.DQNProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                         state_size=state_size, model_params=model_params,
                                         saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent.agent_name == agent_globals.names["ddqn"]:
        problem = dqn_problem.DQNProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                         state_size=state_size, model_params=model_params,
                                         saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent.agent_name == agent_globals.names["dddqn"]:
        problem = dqn_problem.DQNProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                         state_size=state_size, model_params=model_params,
                                         saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent.agent_name == agent_globals.names["dpg"]:
        problem = dpg_problem.DPGProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                         state_size=state_size, model_params=model_params,
                                         saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent.agent_name == agent_globals.names["ddpg"]:
        problem = ddpg_problem.DDPGPRoblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                           state_size=state_size, model_params=model_params,
                                           saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent.agent_name == agent_globals.names["a2c_discrete"]:
        problem = a2c_problem.A2CProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                         state_size=state_size, model_params=model_params,
                                         saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent.agent_name == agent_globals.names["a2c_continuous"]:
        problem = a2c_problem.A2CProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                         state_size=state_size, model_params=model_params,
                                         saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent.agent_name == agent_globals.names["a2c_discrete_queue"]:
        problem = a2c_problem.A2CProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                         state_size=state_size, model_params=model_params,
                                         saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent.agent_name == agent_globals.names["a2c_continuous_queue"]:
        problem = a2c_problem.A2CProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                         state_size=state_size, model_params=model_params,
                                         saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent == "A3C_continuous":
        problem = a3c_problem.A3CProblem(environment, agent=agent, img_input=img_input, n_stack=n_stack,
                                         state_size=state_size, model_params=model_params,
                                         saving_model_params=saving_model_params, net_architecture=net_architecture)

    elif agent == "A3C_discrete":
        problem = a3c_problem.A3CProblem(environment, agent=agent, img_input=img_input, n_stack=n_stack,
                                         state_size=state_size, model_params=model_params,
                                         saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent.agent_name == agent_globals.names["ppo_continuous"]:
        problem = ppo_problem_continuous.PPOProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                                    state_size=state_size, model_params=model_params,
                                                    saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent.agent_name == agent_globals.names["ppo_continuous_parallel"]:
        problem = ppo_problem_continuous_parallel.PPOProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                                             state_size=state_size, model_params=model_params,
                                                             saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent.agent_name == agent_globals.names["ppo_discrete"]:
        problem = ppo_problem_discrete.PPOProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                                  state_size=state_size, model_params=model_params,
                                                  saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent.agent_name == agent_globals.names["ppo_discrete_parallel"]:
        problem = ppo_problem_discrete_parallel.PPOProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                                           state_size=state_size, model_params=model_params,
                                                           saving_model_params=saving_model_params, net_architecture=net_architecture)
    return problem
