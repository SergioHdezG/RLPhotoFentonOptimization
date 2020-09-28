import gym
from gym.utils import play
from IRL_Problem.base.utils.callbacks import Callbacks, load_expert_memories
from RL_Agent import ppo_agent_discrete
from utils import hyperparameters as params
from RL_Problem import rl_problem as rl_p
from IRL_Problem.gail import GAIL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


exp_dir = "../expert_demonstrations/"
exp_name = "MountainCar-v0"

env_name ="MountainCar-v0"

env = gym.make(env_name)

cb = Callbacks()

# A continuación vas a jugar al MountainCar para ser el experto de referencia del agente que aprenderá por IRL
# Para mover el cochecito hay que usar las teclas, para finalizar de grabar experiencias pulsar escape
play.play(env, zoom=2, callback=cb.remember_callback)
env.close()

cb.memory_to_pkl(exp_dir, exp_name)


agent = ppo_agent_discrete.Agent()

model_params = params.algotirhm_hyperparams(learning_rate=1e-3,
                                            batch_size=64,
                                            epsilon=.7,
                                            epsilon_decay=0.99999,
                                            epsilon_min=0.15,
                                            n_step_return=10)

def lstm_custom_model(input_shape):
    actor_model = Sequential()
    actor_model.add(LSTM(64, input_shape=input_shape, activation='tanh'))
    actor_model.add(Dense(128, input_shape=input_shape, activation='relu'))
    actor_model.add(Dense(128, activation='relu'))
    return actor_model

net_architecture = params.actor_critic_net_architecture(use_custom_network=True,
                                                        actor_custom_network=lstm_custom_model,
                                                        critic_custom_network=lstm_custom_model
                                                        )

rl_problem = rl_p.Problem(env_name, agent, model_params, net_architecture=net_architecture, n_stack=4)


discriminator_stack = 4
exp_memory = load_expert_memories(exp_dir, exp_name, load_action=True, n_stack=discriminator_stack)

irl_problem = GAIL(rl_problem, exp_memory, n_stack_disc=discriminator_stack)

irl_problem.solve(200, render=False, render_after=190)
rl_problem.test(10, True)
