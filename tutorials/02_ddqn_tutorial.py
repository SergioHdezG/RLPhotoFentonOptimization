import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from RL_Problem import rl_problem
from RL_Agent import ddqn_agent
from utils import hyperparameters as params

environment = "CartPole-v1"

agent = ddqn_agent.Agent()

model_params = params.algotirhm_hyperparams(learning_rate=1e-3,
                                            batch_size=128,
                                            epsilon=0.4,
                                            epsilon_decay=0.999,
                                            epsilon_min=0.15)

# Se puede definir una red rápida mediante un diccionario o usanto las utilidades de utils.hyperparameters
"""
net_architecture = params.net_architecture(dense_layers=2,
                                           n_neurons=[128, 128],
                                           dense_activation=['relu', 'tanh'])
"""

net_architecture =  {"dense_lay": 2,
                    "n_neurons": [128, 128],
                    "dense_activation": ['relu', 'tanh'],
                    }

# Se debe pasar al problema la arquitectura diseñada para la red y además se va a aplicar un apilado de obsevaciones
# para conseguir que el modelo procese información tempora, esto se indica mediante n_stack=10 de forma que la red
# neuronal verá cada vez las 10 última observaciones (o estado). De esta forma la red neuronal sabe que está ocurriendo
# en el instante de tiempo actual y en los 9 anteriores.
problem = rl_problem.Problem(environment, agent, model_params, net_architecture=net_architecture, n_stack=10)

# En este caso se utiliza el atributo skip_states=3. Esto indica que se va a utilizar una técnica para acelerar la
# recolección de experiencias durante el entrenamiento. skip_states=3 indica que cada vez que la red neuronal seleccione
# una acción, está se va a repetir tres veces seguidas únicamente durante el entrenamiento, al test esto no le afecta.
problem.solve(episodes=100, skip_states=4)
problem.test(n_iter=10)