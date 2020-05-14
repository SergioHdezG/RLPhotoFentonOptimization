from CAPORL.RL_Problem.ActorCritic import a2c_problem, a3c_problem, ddpg_problem
from CAPORL.RL_Problem.ValueBased import dqn_problem
from CAPORL.RL_Problem.PolicyBased import dpg_problem
from CAPORL.RL_Agent.DQN_Agent import dddqn_agent, dqn_agent, ddqn_agent
from CAPORL.RL_Agent.DDPG_Agent import ddpg_agent
from CAPORL.RL_Agent.DPGAgent import dpg_agent
from CAPORL.RL_Agent.ActorCritic.A2C_Agent import a2c_agent_continuous, a2c_agent_discrete_queue, \
    a2c_agent_continuous_queue, a2c_agent_discrete
from src.IRL.IRL_Problem.irl_problem_super import IRLProblemSuper


def Problem(environment, agent_id, exper_agent_id, model_params, saving_model_params, n_stack=1, img_input=False, state_size=None):
    """ Class for selecting an algorithm to use
    :param environment: Environment selected.
    :param agent:       String. ID of the agent to use.
    :param n_stack:     Int >= 1. If 1, there is no stacked input. Number of time related input stacked.
    :param img_input:   Bool. If True, input data is an image.
    :param state_size:  None, Int or Tuple. State dimensions. If None it will be calculated automatically. Int or
                        Tuple format will be useful when preprocessing change the input dimensions.
    :return:
    """
    if agent_id == 'DQN':
        agent = dqn_agent
        rl_problem = dqn_problem.DQNProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                            state_size=state_size, model_params=model_params,
                                            saving_model_params=saving_model_params)
    elif agent_id == 'DDQN':
        agent = ddqn_agent
        rl_problem = dqn_problem.DQNProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                            state_size=state_size, model_params=model_params,
                                            saving_model_params=saving_model_params)
    elif agent_id == 'DDDQN':
        agent = dddqn_agent
        rl_problem = dqn_problem.DQNProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                            state_size=state_size, model_params=model_params,
                                            saving_model_params=saving_model_params)
    elif agent_id == 'DPG':
        agent = dpg_agent
        rl_problem = dpg_problem.DPGProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                            state_size=state_size, model_params=model_params,
                                            saving_model_params=saving_model_params)
    elif agent_id == 'DDPG':
        agent = ddpg_agent
        rl_problem = ddpg_problem.DDPGPRoblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                              state_size=state_size, model_params=model_params,
                                              saving_model_params=saving_model_params)
    elif agent_id == 'A2C_discrete':
        agent = a2c_agent_discrete
        rl_problem = a2c_problem.A2CProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                            state_size=state_size, model_params=model_params,
                                            saving_model_params=saving_model_params)
    elif agent_id == "A2C_continuous":
        agent = a2c_agent_continuous
        rl_problem = a2c_problem.A2CProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                            state_size=state_size, model_params=model_params,
                                            saving_model_params=saving_model_params)
    elif agent_id == 'A2C_discrete_queue':
        agent = a2c_agent_discrete_queue
        rl_problem = a2c_problem.A2CProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                            state_size=state_size, model_params=model_params,
                                            saving_model_params=saving_model_params)
    elif agent_id == "A2C_continuous_queue":
        agent = a2c_agent_continuous_queue
        rl_problem = a2c_problem.A2CProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                            state_size=state_size, model_params=model_params,
                                            saving_model_params=saving_model_params)
    elif agent_id == "A3C_continuous":
        rl_problem = a3c_problem.A3CProblem(environment, agent=agent_id, img_input=img_input, n_stack=n_stack,
                                            state_size=state_size, model_params=model_params,
                                            saving_model_params=saving_model_params)

    elif agent_id == "A3C_discrete":
        rl_problem = a3c_problem.A3CProblem(environment, agent=agent_id, img_input=img_input, n_stack=n_stack,
                                            state_size=state_size, model_params=model_params,
                                            saving_model_params=saving_model_params)

    irl_problem = IRLProblemSuper(environment, rl_problem, img_input=img_input, n_stack=n_stack, state_size=state_size,
                                  saving_model_params=saving_model_params, expert_agent=exper_agent_id)

    return irl_problem
