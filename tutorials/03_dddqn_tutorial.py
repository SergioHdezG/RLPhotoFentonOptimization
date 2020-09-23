from RL_Problem import rl_problem
from RL_Agent.DQN_Agent import dddqn_agent
from utils import hyperparameters as params
from RL_Agent.base.utils.Memory.deque_memory import Memory as deq_m
import numpy as np
import matplotlib.pylab as plt
import gym


environment = "SpaceInvaders-v0"

agent = dddqn_agent.Agent()

model_params = params.algotirhm_hyperparams(learning_rate=1e-3,
                                            batch_size=64,
                                            epsilon=0.9,
                                            epsilon_decay=0.999999,
                                            epsilon_min=0.15)

net_architecture = params.net_architecture(conv_layers=2,
                                           kernel_num=[32, 32],
                                           kernel_size=[3, 3],
                                           kernel_strides=[2, 2],
                                           conv_activation=['relu', 'relu'],
                                           dense_layers=2,
                                           n_neurons=[256, 128],
                                           dense_activation=['relu', 'relu'])

# Función para preprocesar las imágenes
def atari_preprocess(obs):
    # Crop and resize the image
    obs = obs[20:200:2, ::2]

    # Convert the image to greyscale
    obs = obs.mean(axis=2)

    # normalize between from 0 to 1
    obs = obs / 255.
    obs = obs[:, :, np.newaxis]
    return obs

# Guardamos las dimensiones del estado una vez preprocesado, es necesario que el tercer eje marque el número de canales
state_size = (90, 80, 1)

# Diseñamos una función para normalizar o cortar el valor de recompensa original
def clip_norm_atari_reward(rew):
    return np.clip(np.log(1+rew), -1, 1)

aux_env = gym.make(environment)


aux_obs = aux_env.reset()
aux_prep_obs = atari_preprocess(aux_obs)

plt.figure(1)
plt.subplot(121)
plt.imshow(aux_obs)
plt.subplot(122)
plt.imshow(aux_prep_obs.reshape(90, 80))
plt.show()

# Al realizar un preprocesado externo al entorno que modifica las dimensiones originales de las observaciones es
# necesario indicarlo explicitamente en el atributo state_size=(90, 80, 1)
problem = rl_problem.Problem(environment, agent, model_params, net_architecture=net_architecture, n_stack=4,
                             img_input=True, state_size=state_size)

# Indicamos que se quiere usar la función de recompensa y la normalización
problem.preprocess = atari_preprocess
problem.clip_norm_reward = clip_norm_atari_reward

# Seleccionamos el tamaño de la memoria
memory_max_len = 10000  # Indicamos la capacidad máxima de la memoria
problem.agent.set_memory(deq_m, memory_max_len)


# Se selecciona no renderizar hasta el peisodio 8 para accelerar la simulación
problem.solve(render=False, episodes=1000, skip_states=4, render_after=490)
problem.test(n_iter=100, render=True)
