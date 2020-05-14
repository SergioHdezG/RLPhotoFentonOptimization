from CAPORL.RL_Problem.ActorCritic import a3c_problem, ddpg_problem, a2c_problem
from CAPORL.RL_Problem.ValueBased import dqn_problem
from CAPORL.RL_Problem.PolicyBased import dpg_problem
from CAPORL.RL_Problem.PPO import ppo_problem_v1, ppo_problem_v2, ppo_problem_tf, ppo_problem_async, ppo_problem_discrete, ppo_problem_discrete_async
from CAPORL.RL_Agent.DQN_Agent import ddqn_agent, dddqn_agent, dqn_agent
from CAPORL.RL_Agent.DDPG_Agent import ddpg_agent
from CAPORL.RL_Agent.DPGAgent import dpg_agent
from CAPORL.RL_Agent.PPO import ppo_agent_v1, ppo_agent_v2, ppo_agent_tf, ppo_agent_async, ppo_agent_discrete, ppo_agent_discrete_async
from CAPORL.RL_Agent.ActorCritic.A2C_Agent import a2c_agent_continuous, a2c_agent_discrete, a2c_agent_discrete_queue, a2c_agent_continuous_queue

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
    if agent == 'DQN':
        agent = dqn_agent
        problem = dqn_problem.DQNProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                         state_size=state_size, model_params=model_params,
                                         saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent == 'DDQN':
        agent = ddqn_agent
        problem = dqn_problem.DQNProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                         state_size=state_size, model_params=model_params,
                                         saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent == 'DDDQN':
        agent = dddqn_agent
        problem = dqn_problem.DQNProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                         state_size=state_size, model_params=model_params,
                                         saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent == 'DPG':
        agent = dpg_agent
        problem = dpg_problem.DPGProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                         state_size=state_size, model_params=model_params,
                                         saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent == 'DDPG':
        agent = ddpg_agent
        problem = ddpg_problem.DDPGPRoblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                           state_size=state_size, model_params=model_params,
                                           saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent == 'A2C_discrete':
        agent = a2c_agent_discrete
        problem = a2c_problem.A2CProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                         state_size=state_size, model_params=model_params,
                                         saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent == "A2C_continuous":
        agent = a2c_agent_continuous
        problem = a2c_problem.A2CProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                         state_size=state_size, model_params=model_params,
                                         saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent == 'A2C_discrete_queue':
        agent = a2c_agent_discrete_queue
        problem = a2c_problem.A2CProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                         state_size=state_size, model_params=model_params,
                                         saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent == "A2C_continuous_queue":
        agent = a2c_agent_continuous_queue
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
    elif agent == 'PPO_continuous':
        agent = ppo_agent_v2
        problem = ppo_problem_v2.PPOProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                            state_size=state_size, model_params=model_params,
                                            saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent == 'PPO_continuous_async':
        agent = ppo_agent_async
        problem = ppo_problem_async.PPOProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                               state_size=state_size, model_params=model_params,
                                               saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent == 'PPO_discrete':
        agent = ppo_agent_discrete
        problem = ppo_problem_discrete.PPOProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                            state_size=state_size, model_params=model_params,
                                            saving_model_params=saving_model_params, net_architecture=net_architecture)
    elif agent == 'PPO_discrete_async':
        agent = ppo_agent_discrete_async
        problem = ppo_problem_discrete_async.PPOProblem(environment, agent, img_input=img_input, n_stack=n_stack,
                                            state_size=state_size, model_params=model_params,
                                            saving_model_params=saving_model_params, net_architecture=net_architecture)
    return problem
