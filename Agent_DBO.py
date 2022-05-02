import os
import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from RL_Agent import ppo_opt_cicles_agent_continuous_parallel
from utils.preprocess import *
from utils import hyperparameters as hyperparams
from RL_Problem import rl_problem
import global_enviroment_log as glob
import datetime as dt
import yaml
from src.environments import perox_complete_model_small_glob_coord_opt_sec_init
from src.environments import perox_sqp_globals
import matplotlib.pyplot as plt


exp_folder = sys.argv[1]
with open(exp_folder + 'exp_config.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    print(data)

glob.test = data['test']
glob.traj_path = data['experiment_path']
sys.stdout = open(glob.traj_path + 'log_experiment.txt', 'w')

hora_inicio = dt.datetime.now()
print('Start time', hora_inicio)

log_alpha_values_init = [1e-5, 1e-5, 1e-5, -15., 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1, 1]

f = open(data['sodis_params'], "r")
sodis_params = f.read()
sodis_params = sodis_params.split('\n')
sodis_params = [float(sodis_params[i]) for i in range(len(sodis_params)-1)]

if data['optimize_bacteria']:
    f = open(data['perox_params'], "r")
    perox_params = f.read()
    perox_params = perox_params.split('\n')
    perox_params = [float(perox_params[i]) for i in range(len(perox_params) - 1)]

# Assign Sodis params
log_alpha_values_init[6] = sodis_params[0]
log_alpha_values_init[10] = sodis_params[1]
log_alpha_values_init[13] = np.trunc(sodis_params[2])

# Assign Perox params
if data['optimize_bacteria']:
    log_alpha_values_init[0] = perox_params[0]
    log_alpha_values_init[1] = perox_params[1]
    log_alpha_values_init[2] = perox_params[2]
    log_alpha_values_init[4] = perox_params[3]
    log_alpha_values_init[5] = perox_params[4]

if data['optimize_peroxide']:
    m_cinetico = perox_complete_model_small_glob_coord_opt_sec_init.env(sodis=False, perox=True, bact=False,
                                                                 log_alpha_values_init=log_alpha_values_init,
                                                                 log_scale=False)

if data['optimize_bacteria']:
    m_cinetico = perox_complete_model_small_glob_coord_opt_sec_init.env(sodis=False, perox=False, bact=True,
                                                                 log_alpha_values_init=log_alpha_values_init,
                                                                log_scale=False)

def lstm_custom_model_actor(input_shape):
    actor_model = Sequential()
    actor_model.add(LSTM(128, input_shape=input_shape, activation='tanh'))
    actor_model.add(Dense(256, input_shape=input_shape, activation='relu'))
    actor_model.add(Dense(256, input_shape=input_shape, activation='relu'))
    actor_model.add(Dense(128, activation='relu'))
    actor_model.add(Dense(5, activation='tanh'))
    return actor_model

def lstm_custom_model_critic(input_shape):
    critic_model = Sequential()
    critic_model.add(LSTM(128, input_shape=input_shape, activation='tanh'))
    critic_model.add(Dense(256, input_shape=input_shape, activation='relu'))
    critic_model.add(Dense(128, activation='relu'))
    critic_model.add(Dense(1, activation='linear'))

    return critic_model

net_architecture = hyperparams.actor_critic_net_architecture(use_custom_network=True,
                                                        actor_custom_network=lstm_custom_model_actor,
                                                        critic_custom_network=lstm_custom_model_critic,
                                                        define_custom_output_layer=True)

agent = ppo_opt_cicles_agent_continuous_parallel.Agent(actor_lr=float(data['actor_lr']),
                                            critic_lr=float(data['critic_lr']),
                                            batch_size=data['batch_size'],
                                            exploration_noise=data['exploration_noise'],
                                            epsilon=data['epsilon'],
                                            epsilon_decay=data['epsilon_decay'],
                                            epsilon_min=data['epsilon_min'],
                                            memory_size=data['memory_size'],
                                            net_architecture=net_architecture,
                                            n_stack=data['n_stack'],
                                            n_step_return=20,
                                            histogram_memory=True,
                                            tensorboard_dir=None,
                                            n_parallel_envs=data["n_threads"]
                                            )

perox_sqp_globals.n_stack = data['n_stack']
perox_sqp_globals.n_parallel_envs = agent.n_parallel_envs
problem = rl_problem.Problem(m_cinetico, agent)

if data['optimize_peroxide']:
    problem.preprocess = perox_only_norm
if data['optimize_bacteria']:
    problem.preprocess = bact_only_norm

problem.compile()

iter_init = dt.datetime.now()

problem.solve(data['iter'], render=False)
opt_time = dt.datetime.now() - iter_init

opt_end_hour = dt.datetime.now()

best_params = []
best_parallel_params_list = problem.env.get_best_params()
for parall_list in best_parallel_params_list:
    for params in parall_list:
        try:
            string = ''
            for i in range(params[0].shape[0]):
                string += str(params[0][i]) + "\t"
            best_params.append(string + str(params[1]) + "\t" + str(int((params[2]/2)*12)) + "\n")
        except:
            pass

best = 1000.
best_str = ''
for param in glob.best_params_text_list:
    error = param.split('\t')[3]
    error = float(error[:-2])
    if best > error:
        best = error
        best_str = param

print(best_str)

best_parallel_params_list = problem.env.get_best_params()

params_list = []
for params in best_parallel_params_list:
    params_list.extend(params)

if len(params_list) > 0:
    ind_best = np.argmin(np.array(params_list)[:, 1])
    model_params, model_error, iteration = params_list[ind_best][0], params_list[ind_best][1], params_list[ind_best][2]
else:
    model_params = 'No params'
    model_error = 'No error'
    iteration = ' No data'

print("Model params: ", str(model_params), " Model error: ", str(model_error), " Iter: ", str(int((iteration/2)*12)))
print('Hora inicio', hora_inicio)
print('Opt Time: ', opt_time)
print('End time', opt_end_hour)

# problem.test(render=True, n_iter=data["test_iter"]

print('Hora inicio', hora_inicio)
print('Perox Time: ', opt_time)


if glob.test:
    for i in range(len(glob.traj_params_to_write)):
        file = open(glob.traj_path + 'trajectories_' + str(i) + '.txt', "a")
        file.writelines(glob.traj_params_to_write[i])
        file.close()

# Print sodis best parameters
print("Perox params: ", str(model_params), " Model error: ", str(model_error), " Iter: ", str(int((iteration/2)*12)))

# Print best parameters
best_params = []
for parall_list in best_parallel_params_list:
    for params in parall_list:
        try:
            string = ""
            for p in params[0]:
                string += p + "\t"
            string += params[1] + "\t" + str(int((params[2]/2)*12)) + "\n"
            best_params.append(string)

        except:
            pass

path = os.path.abspath(glob.traj_path + 'best_params.txt')
file = open(path, "a")
file.writelines(best_params)
file.writelines(glob.best_params_text_list)
file.close()

print('Hora inicio', hora_inicio)
print('Perox Time: ', opt_time)
print('End time', opt_end_hour)

if data['optimize_peroxide']:
    f = open(glob.traj_path + "peroxide_params.txt", "w")
if data['optimize_bacteria']:
    f = open(glob.traj_path + "bacteria_params.txt", "w")

for p in model_params:
    f.write(str(p)+"\n")
f.write(str(model_error)+"\n")
f.close()

from src.environments.fotocaos_complete_model import modelo_completo_perox_interfaz
model = modelo_completo_perox_interfaz.ModeloCinetico(perox=True, bact=True, sodis=True,
                                                      params_to_optimize=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
log_alpha_values_init[0] = model_params[0]
log_alpha_values_init[1] = model_params[1]
log_alpha_values_init[2] = model_params[2]
log_alpha_values_init[4] = model_params[3]
log_alpha_values_init[5] = model_params[4]
params = log_alpha_values_init
error, curvas, params = model.reset(params=np.array(params))
model.render(error, curvas, params, 0., perox=True, bact=True, sodis=True)
plt.savefig(glob.traj_path + 'model_fitted.png')

sys.stdout.close
