import random

from CAPORL.RL_Problem import rl_problem
from CAPORL.RL_Agent.ActorCritic.A2C_Agent import a2c_agent_continuous, a2c_agent_discrete, a2c_agent_discrete_queue, a2c_agent_continuous_queue
from CAPORL.RL_Agent.ActorCritic.A3C_Agent import a3c_agent_continuous, a3c_agent_discrete
from CAPORL.RL_Agent.DPGAgent import dpg_agent
from CAPORL.RL_Agent.DDPG_Agent import ddpg_agent
from CAPORL.RL_Agent.DQN_Agent import dqn_agent, ddqn_agent, dddqn_agent
from CAPORL.RL_Agent.PPO import ppo_agent_async, ppo_agent_v2
from CAPORL.utils.clipping_reward import *
from CAPORL.utils.preprocess import *
from CAPORL.utils import hyperparameters as params
from CAPORL.Memory.deque_memory import Memory as deque_m
from src.IRL.utils import callbacks
from CAPORL.utils.custom_networks import custom_nets
from CAPORL.environments import carlaenv_continuous, carlaenv_continuous_stop, carlaenv_continuous_rl


# environment = "LunarLanderContinuous-v2"
# environment = el_viajante
environment = carlaenv_continuous_rl.env

# agent = ppo_agent_async.create_agent()
agent = ppo_agent_v2.create_agent()

model_params = params.algotirhm_hyperparams(learning_rate=1e-4,
                                            batch_size=128,
                                            epsilon=0.5,
                                            epsilon_decay=0.99995,
                                            epsilon_min=0.15,
                                            n_step_return=10)

# net_architecture = params.net_architecture(conv_layers=3,
#                                            kernel_num=[32, 32, 64],
#                                            kernel_size=[7, 5, 3],
#                                            kernel_strides=[4, 2, 2],
#                                            conv_activation=['relu', 'relu', 'relu'],
#                                            dense_layers=2,
#                                            n_neurons=[32, 32],
#                                            dense_activation=['relu', 'relu'])

net_architecture = params.actor_critic_net_architecture(actor_conv_layers=2, actor_kernel_num=[32, 32],
                                                        actor_kernel_size=[7, 5], actor_kernel_strides=[4, 2],
                                                        actor_conv_activation=['relu', 'relu'],
                                                        actor_dense_layers=3, actor_n_neurons=[256, 256, 128],
                                                        actor_dense_activation=['relu', 'relu', 'relu'],
                                                        critic_conv_layers=2, critic_kernel_num=[32, 32],
                                                        critic_kernel_size=[7, 5], critic_kernel_strides=[4, 2],
                                                        critic_conv_activation=['relu', 'relu'],
                                                        critic_dense_layers=3, critic_n_neurons=[256, 256, 128],
                                                        critic_dense_activation=['relu', 'relu', 'relu'],
                                                        use_custom_network=True,
                                                        actor_custom_network=custom_nets.actor_model_lstm,
                                                        critic_custom_network=custom_nets.critic_model_lstm
                                                        )

# net_architecture = None

saving_model_params = params.save_hyperparams(base_dir="saved_models/Carla/RL_1/1/",
                                              model_name="PPO_IRL_carla",
                                              save_each=10,
                                              save_if_better=False)


# saving_model_params = None

state_size = None
n_stack = 10
img_input = False

problem = rl_problem.Problem(environment, agent, model_params, saving_model_params, net_architecture=net_architecture,
                             n_stack=n_stack, img_input=img_input, state_size=state_size)

# memory_max_len = 50000  # Indicamos la capacidad máxima de la memoria
# problem.agent.set_memory(deque_m, memory_max_len)

# problem.preprocess = atari_assault_preprocess
# problem.preprocess = preproces_car_racing
# problem.clip_reward = clip_reward_atari
dir_load="/home/serch/TFM/IRL3/saved_models/Carla/RL_1/1/"
name_loaded="PPO_IRL_carla174-69"


# callback = callbacks.Callbacks()
problem.load_model(dir_load=dir_load, name_loaded=name_loaded)
# problem.solve(100, render=False, max_step_epi=1000, render_after=3990, skip_states=1)
# problem.test(render=True, n_iter=400, callback=callback.remember_callback)
problem.test(render=True, n_iter=400, callback=None)

# callback.memory_to_csv('expert_demonstrations/', 'Expert_PPO_LunarLanderContinuous')