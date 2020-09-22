from CAPORL.RL_Problem.ActorCritic import a3c_problem, ddpg_problem, a2c_problem
from CAPORL.RL_Problem.ValueBased import dqn_problem
from CAPORL.RL_Problem.PolicyBased import dpg_problem
from CAPORL.RL_Problem.PPO import ppo_problem_continuous, ppo_problem_continuous_parallel, ppo_problem_discrete, ppo_problem_discrete_parallel
from CAPORL.RL_Agent.DQN_Agent import ddqn_agent, dddqn_agent, dqn_agent
from CAPORL.RL_Agent.DDPG_Agent import ddpg_agent
from CAPORL.RL_Agent.DPG_Agent import dpg_agent
from CAPORL.RL_Agent.PPO import ppo_agent_continuous, ppo_agent_continuous_parallel, ppo_agent_discrete, ppo_agent_discrete_parallel
from CAPORL.RL_Agent.ActorCritic.A2C_Agent import a2c_agent_continuous, a2c_agent_discrete, a2c_agent_discrete_queue, a2c_agent_continuous_queue
from CAPORL.RL_Agent.base.utils import agent_globals


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
