from CAPORL.RL_Agent.DQN_Agent import dddqn_agent, dqn_agent
from CAPORL.RL_Agent.DPGAgent import dpg_agent
from CAPORL.RL_Problem import rl_problem as rl_p
from CAPORL.utils.clipping_reward import *
from CAPORL.utils.preprocess import *
from CAPORL.utils import hyperparameters as params
from CAPORL.environments import carlaenv_continuous
from src.IRL.IRL_Problem import irl_problem_super as irl_p
from src.IRL.utils.callbacks import load_expert_memories
from CAPORL.RL_Agent.PPO import ppo_agent_async, ppo_agent_v2, ppo_agent_discrete
from CAPORL.environments import CarRacing
from CAPORL.utils.custom_networks import custom_nets


# environment = "LunarLanderContinuous-v2"
environment = CarRacing.env

agent = ppo_agent_v2.create_agent()

model_params = params.algotirhm_hyperparams(learning_rate=1e-3,
                                            batch_size=128,
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
                                                        actor_custom_network=custom_nets.actor_model_drop,
                                                        critic_custom_network=custom_nets.critic_model_drop
                                                        )

saving_model_params = params.save_hyperparams(base_dir="saved_models/LunarLander/1/",
                                              model_name="LunarLander_DPG_IRL",
                                              save_each=200,
                                              save_if_better=True)

saving_model_params = None

# Loading expert memories
exp_dir = "expert_demonstrations/"
exp_name = 'human_expert_CarRacing_v2'
# exp_name = 'Expert_PPO_LunarLanderContinuous'
disc_stack = 4
exp_memory = load_expert_memories(exp_dir, exp_name, load_action=True, n_stack=disc_stack)

state_size=None
n_stack = 4
img_input = False

rl_problem = rl_p.Problem(environment, agent, model_params, saving_model_params, net_architecture=net_architecture,
                             n_stack=n_stack, img_input=img_input, state_size=state_size)

dir_load="saved_experts/CarRacing/"
name_loaded="CarRacing_bc-131"
# rl_problem.load_model(dir_load=dir_load, name_loaded=name_loaded)

irl_problem = irl_p.Problem(rl_problem, exp_memory, stack_disc=disc_stack > 1)


irl_problem.solve(2000, render=False, max_step_epi=None, render_after=1500, skip_states=1)
