import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from RL_Problem import rl_problem
from IRL_Problem.gail import GAIL
from RL_Agent import ppo_agent_discrete_parallel
from utils import hyperparameters as params
from IRL_Problem.base.utils.callbacks import load_expert_memories
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

environment = "CartPole-v1"
# environment = el_viajante

exp_dir = "../expert_demonstrations/"
exp_name="Expert_CartPole"

# En este ejemplo se provee el archivo Expert_CartPole.pkl en la carpeta expert_demonstations que contiene las
# experiencias recogidas de un agente experto entrenado mediante RL. Si se desea repetir el proceso completo y
# recolectar de nuevo las experiencias del agente solo es necesario descomentar la siguiente parte del código.
"""
expert = dpg_agent.Agent()

model_params = params.algotirhm_hyperparams(learning_rate=5e-4,
                                            batch_size=32,
                                            epsilon=0.6,
                                            epsilon_decay=0.99999,
                                            epsilon_min=0.1,
                                            n_step_return=15)

net_architecture = params.net_architecture(dense_layers=2,
                                           n_neurons=[256, 256],
                                           dense_activation=['relu', 'relu'])

# net_architecture = params.actor_critic_net_architecture(actor_conv_activation=['relu', 'relu'],
#                                                         actor_dense_layers=2, actor_n_neurons=[256, 256],
#                                                         actor_dense_activation=['relu', 'relu'],
#                                                         critic_dense_layers=2, critic_n_neurons=[256, 256],
#                                                         critic_dense_activation=['relu', 'relu'])

# net_architecture = None

saving_model_params = params.save_hyperparams(base_dir="saved_experts/LunarLander/1/",
                                              model_name="LunarLander_ddqn",
                                              save_each=100,
                                              save_if_better=True)

saving_model_params = None

state_size = None

expert_problem = rl_problem.Problem(environment, expert, model_params, saving_model_params, net_architecture=net_architecture,
                             n_stack=1, img_input=False, state_size=state_size)


callback = Callbacks()
# problem.load_model(dir_load=dir_load, name_loaded=name_loaded)
expert_problem.solve(1000, render=False, max_step_epi=None, render_after=3990, skip_states=3)
expert_problem.test(render=False, n_iter=200, callback=callback.remember_callback)

callback.memory_to_pkl(exp_dir, exp_name)
"""


agent = ppo_agent_discrete_parallel.Agent()

model_params = params.algotirhm_hyperparams(learning_rate=1e-3,
                                            batch_size=32)


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

rl_problem = rl_problem.Problem(environment, agent, model_params, net_architecture=net_architecture, n_stack=3)


discriminator_stack = 3
exp_memory = load_expert_memories(exp_dir, exp_name, load_action=True, n_stack=discriminator_stack)

irl_params = params.irl_hyperparams(lr_disc=1e-4, batch_size_disc=128, epochs_disc=5, val_split_disc=0.2,
                                    agent_collect_iter=15, agent_train_iter=100)

def one_layer_custom_model(input_shape):
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape, activation='relu'))
    model.add(Dense(128, input_shape=input_shape, activation='relu'))
    return model


irl_net_architecture = params.irl_discriminator_net_architecture(use_custom_network=True,
                                                                 state_custom_network=None,
                                                                 action_custom_network=None,
                                                                 common_custom_network=one_layer_custom_model,
                                                                 last_layer_activation='sigmoid')

irl_problem = GAIL(rl_problem, exp_memory, n_stack_disc=discriminator_stack, irl_params=irl_params,
                      net_architecture=irl_net_architecture)

irl_problem.solve(1500, render=False, max_step_epi=None, render_after=1500, skip_states=1)
rl_problem.test(10)