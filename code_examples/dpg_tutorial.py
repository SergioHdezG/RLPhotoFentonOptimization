from CAPORL.RL_Problem import rl_problem
from CAPORL.RL_Agent.DPGAgent import dpg_agent
from CAPORL.utils import hyperparameters as params
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


environment = "LunarLander-v2"

agent = dpg_agent.create_agent()

model_params = params.algotirhm_hyperparams(learning_rate=1e-3,
                                            batch_size=64)

# Para definir un red con arquitecturas expeciales que vayan más allá de capas convolucionales y densas se debe crear
# una función que defina la arquitectura de la red sin especificar la última capa que será una capa densa con número de
# neuronas correspondiente al número de acciones
def lstm_custom_model(input_shape):
    actor_model = Sequential()
    actor_model.add(LSTM(64, input_shape=input_shape, activation='tanh'))
    actor_model.add(Dense(256, activation='relu'))
    actor_model.add(Dense(256, activation='relu'))
    return actor_model

# Despues es necesario crear un diccionario indicando que se va a usar una red custom y su arquitectura definida antes
net_architecture = params.net_architecture(use_custom_network=True,
                                           custom_network=lstm_custom_model)

problem = rl_problem.Problem(environment, agent, model_params, net_architecture=net_architecture, n_stack=5)

# En este caso no se expecifica ningun tipo de memoria ya que no aplica a este algoritmo

# Se selecciona no renderizar hasta el peisodio 190 para accelerar la simulación
# Al seleccionar skip_states la renderización durante el entrenamiento se ve accelerada
problem.solve(render=False, episodes=200, skip_states=3, render_after=190)
problem.test(n_iter=10, render=True)
