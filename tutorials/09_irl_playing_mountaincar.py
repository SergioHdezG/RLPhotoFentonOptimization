import gym
from gym.utils import play
from src.IRL.utils.callbacks import Callbacks, load_expert_memories
from CAPORL.RL_Agent.PPO import ppo_agent_discrete
from CAPORL.utils import hyperparameters as params
from CAPORL.RL_Problem import rl_problem as rl_p
from src.IRL.IRL_Problem.base import irl_problem_super as irl_p
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


exp_dir = "../expert_demonstrations/"
exp_name = "MountainCar-v0"

env_name ="MountainCar-v0"

env = gym.make(env_name)

cb = Callbacks()

play.play(env, zoom=3, callback=cb.remember_callback)

cb.memory_to_pkl(exp_dir, exp_name)


agent = ppo_agent_discrete.create_agent()

model_params = params.algotirhm_hyperparams(learning_rate=1e-3,
                                            batch_size=64,
                                            epsilon=.7,
                                            epsilon_decay=0.99999,
                                            epsilon_min=0.15,
                                            n_step_return=10)

def lstm_custom_model(input_shape):
    actor_model = Sequential()
    actor_model.add(LSTM(64, input_shape=input_shape, activation='tanh'))
    actor_model.add(Dense(256, input_shape=input_shape, activation='relu'))
    actor_model.add(Dense(256, activation='relu'))
    return actor_model

net_architecture = params.actor_critic_net_architecture(use_custom_network=True,
                                                        actor_custom_network=lstm_custom_model,
                                                        critic_custom_network=lstm_custom_model
                                                        )

# net_architecture = params.net_architecture(use_custom_network=True,
#                                            custom_network=lstm_custom_model)

rl_problem = rl_p.Problem(env_name, agent, model_params, net_architecture=net_architecture, n_stack=4)


discriminator_stack = 4
exp_memory = load_expert_memories(exp_dir, exp_name, load_action=True, n_stack=discriminator_stack)

irl_problem = irl_p.IRLProblemSuper(rl_problem, exp_memory, n_stack_disc=True)

irl_problem.solve(500, render=False, max_step_epi=None, render_after=1500, skip_states=1)
rl_problem.test(10, True)