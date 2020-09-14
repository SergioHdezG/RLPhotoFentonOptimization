from CAPORL.Memory.deque_memory import Memory as deq_m
from CAPORL.RL_Problem import rl_problem
from CAPORL.RL_Agent.PPO import ppo_agent_discrete, ppo_agent_v2, ppo_agent_async, ppo_agent_discrete_async
from CAPORL.utils import hyperparameters as params


environment_disc = "LunarLander-v2"
environment_cont = "Pendulum-v0"

# Encontramos cuatro tipos de agentes PPO, dos para problemas con acciones discretas (ppo_agent_discrete,
# ppo_agent_discrete_async) y dos para acciones continuas (ppo_agent_v2, ppo_agent_async). Por
# otro lado encontramos una versión de cada uno siíncrona y otra asíncrona.
agent_disc = ppo_agent_discrete.create_agent()
# agent_disc = ppo_agent_discrete_async.create_agent()
agent_cont = ppo_agent_v2.create_agent()
# agent_cont = ppo_agent_async.create_agent()

# Este algoritmo utiliza el parámetro n_step_return que indica que ventana de tiempo se utiliza para calcular el valor
# del retorno durante la optimización. En este caso una ventana temporal de los 15 últimos estados.
model_params_disc = params.algotirhm_hyperparams(learning_rate=1e-3,
                                            batch_size=64,
                                            epsilon=0.9,
                                            epsilon_decay=0.95,
                                            epsilon_min=0.15)

# En el caso continuo no es necesario especificar los parámetros relacionados con epsilon ya que la aleatoriedad en la
# selección de acciones se realiza muestreando de una distribución normal.
model_params_cont = params.algotirhm_hyperparams(learning_rate=1e-3,
                                                 batch_size=64,
                                                 epsilon=1.0,
                                                 epsilon_decay=0.995,
                                                 epsilon_min=0.15)


# Los algoritmos Ator-Critic utilizan dos redes neronales, una el Actor y otra el Crítico, la forma rápida de crearlas
# es la siguiente (Anunque en este experimento solo se van autilizar capas densas se definen también capas
# convolucionales a modo de ejemplo que luego la librería descartará al crear el modelo ya que el tipo de entrada no se
# corresponde con el necesario para realizar convoluciones. Para que se realizasen tendriamos que marcar el parámetro
# img_input=False al construir el problema más adelante).
net_architecture = params.actor_critic_net_architecture(
                    actor_conv_layers=3,                            critic_conv_layers=2,
                    actor_kernel_num=[32, 64, 32],                  critic_kernel_num=[32, 32],
                    actor_kernel_size=[7, 5, 3],                    critic_kernel_size=[3, 3],
                    actor_kernel_strides=[4, 2, 1],                 critic_kernel_strides=[2, 2],
                    actor_conv_activation=['relu', 'relu', 'relu'], critic_conv_activation=['tanh', 'tanh'],
                    actor_dense_layers=2,                           critic_dense_layers=2,
                    actor_n_neurons=[64, 64],                     critic_n_neurons=[64, 64],
                    actor_dense_activation=['relu', 'relu'],        critic_dense_activation=['relu', 'relu']
                    )

# # Descomentar para ejecutar el ejemplo discreto
# problem_disc = rl_problem.Problem(environment_disc, agent_disc, model_params_disc, net_architecture=net_architecture,
#                              n_stack=1)
#
# # En este caso se utiliza el parámetro max_step_epi=500 para indicar que cada episodio termine a las 500 épocas o
# # iteraciones ya que por defecto este entorno llega hasta 1000. Esto es util para entornos que no tengan definido un
# # máximo de épocas.
# problem_disc.solve(3000, render=False, max_step_epi=500, skip_states=1)
# problem_disc.test(render=True, n_iter=10)


# Descomentar para ejecutar el ejemplo continuo
problem_cont= rl_problem.Problem(environment_cont, agent_cont, model_params_cont, net_architecture=net_architecture,
                             n_stack=1)
# En este caso no se utiliza el parámetro max_step_epi=500 por lo que el máximo de iteraciones será el que viene por
# defecto (1000).
problem_cont.solve(500, render=False, skip_states=1)
problem_cont.test(render=True, n_iter=5)
