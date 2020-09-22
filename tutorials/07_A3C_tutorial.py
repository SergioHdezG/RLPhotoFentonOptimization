from CAPORL.Memory.deque_memory import Memory as deq_m
from CAPORL.RL_Problem import rl_problem
from CAPORL.RL_Agent.ActorCritic.A3C_Agent import a3c_agent_discrete, a3c_agent_continuous
from CAPORL.utils import hyperparameters as params
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import numpy as np

environment_disc = "LunarLander-v2"
environment_disc = "SpaceInvaders-v0"
environment_cont = "LunarLanderContinuous-v2"

# Encontramos dos tipos de agentes A3C, uno para acciones continuas y otro para acciones discretas.
agent_disc = a3c_agent_discrete.create_agent()
agent_cont = a3c_agent_continuous.create_agent()

# Este algoritmo utiliza el parámetro n_step_return que indica que ventana de tiempo se utiliza para calcular el valor
# del retorno durante la optimización. En este caso una ventana temporal de los 15 últimos estados.
model_params_disc = params.algotirhm_hyperparams(learning_rate=1e-3,
                                            batch_size=64,
                                            epsilon=0.9,
                                            epsilon_decay=0.9995,
                                            epsilon_min=0.15,
                                            n_step_return=32)

# En el caso continuo no es necesario especificar los parámetros relacionados con epsilon ya que la aleatoriedad en la
# selección de acciones se realiza muestreando de una distribución normal.
model_params_cont = params.algotirhm_hyperparams(learning_rate=1e-3,
                                                batch_size=64,
                                                n_step_return=16)


# Los algoritmos Ator-Critic utilizan dos redes neronales, una el Actor y otra el Crítico, la forma rápida de crearlas
# es la siguiente (Anunque en este experimento solo se van autilizar capas densas se definen también capas
# convolucionales a modo de ejemplo que luego la librería descartará al crear el modelo ya que el tipo de entrada no se
# corresponde con el necesario para realizar convoluciones. Para que se realizasen tendriamos que marcar el parámetro
# img_input=False al construir el problema más adelante).
net_architecture = params.actor_critic_net_architecture(
                    actor_dense_layers=3,                           critic_dense_layers=3,
                    actor_n_neurons=[128, 256, 128],                     critic_n_neurons=[128, 256, 128],
                    actor_dense_activation=['relu', 'relu', 'relu'],        critic_dense_activation=['relu', 'relu', 'relu']
                    )

def lstm_custom_model(input_shape):
    actor_model = Sequential()
    # actor_model.add(LSTM(64, input_shape=input_shape, activation='tanh'))
    actor_model.add(
        Conv2D(32, kernel_size=3, input_shape=input_shape, strides=2, padding='same', activation='relu'))
    actor_model.add(
        Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))
    actor_model.add(
        Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))
    actor_model.add(Flatten())
    actor_model.add(Dense(1024, activation='relu'))
    actor_model.add(Dense(512, activation='relu'))
    return actor_model

# Despues es necesario crear un diccionario indicando que se va a usar una red custom y su arquitectura definida antes
net_architecture = params.actor_critic_net_architecture(use_custom_network=True,
                                                        actor_custom_network=lstm_custom_model,
                                                        critic_custom_network=lstm_custom_model)


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

# Descomentar para ejecutar el ejemplo discreto
problem_disc = rl_problem.Problem(environment_disc, agent_disc, model_params_disc, net_architecture=net_architecture,
                             n_stack=5, img_input=True, state_size=state_size)

# Indicamos que se quiere usar la función de recompensa y la normalización
problem_disc.preprocess = atari_preprocess
problem_disc.clip_norm_reward = clip_norm_atari_reward

# En este caso se utiliza el parámetro max_step_epi=500 para indicar que cada episodio termine a las 500 épocas o
# iteraciones ya que por defecto este entorno llega hasta 1000. Esto es util para entornos que no tengan definido un
# máximo de épocas.
problem_disc.solve(3000, render=True, skip_states=3)
problem_disc.test(render=True, n_iter=100)


# # Descomentar para ejecutar el ejemplo continuo
# problem_cont= rl_problem.Problem(environment_cont, agent_cont, model_params_cont, net_architecture=net_architecture,
#                              n_stack=4)
# # En este caso no se utiliza el parámetro max_step_epi=500 por lo que el máximo de iteraciones será el que viene por
# # defecto (1000).
# problem_cont.solve(300, render=False, max_step_epi=500, skip_states=1)
# problem_cont.test(render=True, n_iter=10)
