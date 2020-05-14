import random

from CAPORL.RL_Problem import rl_problem
from CAPORL.RL_Agent.ActorCritic.A2C_Agent import a2c_agent_continuous, a2c_agent_discrete, a2c_agent_discrete_queue, a2c_agent_continuous_queue
from CAPORL.RL_Agent.ActorCritic.A3C_Agent import a3c_agent_continuous, a3c_agent_discrete
from CAPORL.RL_Agent.DPGAgent import dpg_agent
from CAPORL.RL_Agent.DDPG_Agent import ddpg_agent
from CAPORL.RL_Agent.DQN_Agent import dqn_agent, ddqn_agent, dddqn_agent
from CAPORL.utils.clipping_reward import *
from CAPORL.utils.preprocess import *
from CAPORL.utils import hyperparameters as params
from CAPORL.Memory.deque_memory import Memory as deque_m
from src.IRL.utils import callbacks
from CAPORL.RL_Agent.PPO import ppo_agent_discrete_async


environment = "LunarLander-v2"
# environment = el_viajante

agent = ppo_agent_discrete_async.create_agent()

model_params = params.algotirhm_hyperparams(learning_rate=1e-4,
                                            batch_size=32,
                                            epsilon=0.6,
                                            epsilon_decay=0.99999,
                                            epsilon_min=0.1,
                                            n_step_return=15)

net_architecture = params.net_architecture(conv_layers=3,
                                           kernel_num=[32, 32, 64],
                                           kernel_size=[7, 5, 3],
                                           kernel_strides=[4, 2, 2],
                                           conv_activation=['relu', 'relu', 'relu'],
                                           dense_layers=2,
                                           n_neurons=[32, 32],
                                           dense_activation=['relu', 'relu'])

net_architecture = params.actor_critic_net_architecture(actor_conv_layers=2, actor_kernel_num=[32, 32],
                                                        actor_kernel_size=[7, 5], actor_kernel_strides=[4, 2],
                                                        actor_conv_activation=['relu', 'relu'],
                                                        actor_dense_layers=2, actor_n_neurons=[256, 256],
                                                        actor_dense_activation=['relu', 'relu'],
                                                        critic_conv_layers=2, critic_kernel_num=[32, 32],
                                                        critic_kernel_size=[7, 5], critic_kernel_strides=[4, 2],
                                                        critic_conv_activation=['relu', 'relu'],
                                                        critic_dense_layers=2, critic_n_neurons=[256, 256],
                                                        critic_dense_activation=['relu', 'relu'])

# net_architecture = None

saving_model_params = params.save_hyperparams(base_dir="saved_experts/LunarLander/1/",
                                              model_name="LunarLander_ddqn",
                                              save_each=100,
                                              save_if_better=True)

saving_model_params = None

state_size = None

problem = rl_problem.Problem(environment, agent, model_params, saving_model_params, net_architecture=net_architecture,
                             n_stack=1, img_input=False, state_size=state_size)

memory_max_len = 50000  # Indicamos la capacidad máxima de la memoria
# problem.agent.set_memory(deque_m, memory_max_len)

# problem.preprocess = atari_assault_preprocess
# problem.preprocess = preproces_car_racing
# problem.clip_reward = clip_reward_atari
dir_load="saved_experts/LunarLander/1/"
name_loaded="LunarLander_ddqn-214"


callback = callbacks.Callbacks()
# problem.load_model(dir_load=dir_load, name_loaded=name_loaded)
problem.solve(700, render=False, max_step_epi=None, render_after=3990, skip_states=1)
problem.test(render=False, n_iter=400, callback=callback.remember_callback)

callback.memory_to_csv('expert_demonstrations/', 'Expert_PPO_LunarLander')