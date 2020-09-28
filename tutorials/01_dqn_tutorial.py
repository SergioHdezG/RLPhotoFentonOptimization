from RL_Problem import rl_problem
from RL_Agent import dqn_agent
from utils import hyperparameters as params

environment = "CartPole-v1"
# environment = gym.make(environment)

agent = dqn_agent.Agent()

model_params = params.algotirhm_hyperparams(learning_rate=1e-3,
                                            batch_size=128,
                                            epsilon=0.4,
                                            epsilon_decay=0.9999,
                                            epsilon_min=0.15)


problem = rl_problem.Problem(environment, agent, model_params)

problem.solve(episodes=100)
problem.test(n_iter=10)
