import os
import datetime
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

from CAPORL.RL_Problem import rl_problem as rl_p
from CAPORL.utils import hyperparameters as params
from CAPORL.environments import carlaenv_continuous
from src.IRL.IRL_Problem.base import irl_problem_super as irl_p
from src.IRL.utils.callbacks import load_expert_memories
from CAPORL.RL_Agent.PPO import ppo_agent_continuous
from CAPORL.utils.custom_networks import custom_nets

# environment = "LunarLanderContinuous-v2"
# environment = CarRacing.env
# environment = carlaenv_continuous_stop.env
environment = carlaenv_continuous.env

agent = ppo_agent_continuous.create_agent()

model_params = params.algotirhm_hyperparams(learning_rate=1e-4,
                                            batch_size=64,
                                            epsilon=.5,
                                            epsilon_decay=0.99995,
                                            epsilon_min=0.15,
                                            n_step_return=10)

net_architecture = params.net_architecture(conv_layers=3,
                                           kernel_num=[32, 32, 64],
                                           kernel_size=[7, 5, 3],
                                           kernel_strides=[4, 2, 2],
                                           conv_activation=['relu', 'relu', 'relu'],
                                           dense_layers=2,
                                           n_neurons=[64, 64],
                                           dense_activation=['relu', 'relu'])

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

saving_model_params = params.save_hyperparams(base_dir="saved_models/steer/2/",
                                              model_name="PPO_IRL_carla",
                                              save_each=100,
                                              save_if_better=False)

# saving_model_params = None

# Loading expert memories
exp_dir = "expert_demonstrations/ultimos/"
# exp_name = 'human_expert_CarRacing_v2'
exp_name = 'human_expert_carla_road_stops'
# exp_name = "human_expert_carla_road_steer"
disc_stack = 4
exp_memory = load_expert_memories(exp_dir, exp_name, load_action=True, n_stack=disc_stack)

state_size=None
n_stack = 4
img_input = False

rl_problem = rl_p.Problem(environment, agent, model_params, saving_model_params, net_architecture=net_architecture,
                             n_stack=n_stack, img_input=img_input, state_size=state_size)


dir_load="saved_models/"
name_loaded="Carla_bc_-10"
dir_load="/home/shernandez/PycharmProjects/IRL3/saved_models/pc/4/"
name_loaded="PPO_IRL_carla146-149"
rl_problem.load_model(dir_load=dir_load, name_loaded=name_loaded)

print(datetime.datetime.now())
irl_problem = irl_p.IRLProblemSuper(rl_problem, exp_memory, n_stack_disc=disc_stack > 1)

# irl_problem.solve(500, render=False, max_step_epi=None, render_after=1500, skip_states=1)
rl_problem.test(10, True)

