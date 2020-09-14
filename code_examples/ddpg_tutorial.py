from CAPORL.Memory.deque_memory import Memory as deq_m
from CAPORL.RL_Problem import rl_problem
from CAPORL.RL_Agent.DDPG_Agent import ddpg_agent
from CAPORL.utils import hyperparameters as params

environment = "MountainCarContinuous-v0"

agent = ddpg_agent.create_agent()

# Este algoritmo utiliza el parámetro n_step_return que indica que ventana de tiempo se utiliza para calcular el valor
# del retorno durante la optimización. En este caso una ventana temporal de los 15 últimos estados.
model_params = params.algotirhm_hyperparams(learning_rate=1e-3,
                                            batch_size=64,
                                            epsilon=0.9,
                                            epsilon_decay=0.9999,
                                            epsilon_min=0.15)


# Los algoritmos Ator-Critic utilizan dos redes neronales, una el Actor y otra el Crítico, la forma rápida de crearlas
# es la siguiente (Anunque en este experimento solo se van autilizar capas densas se definen también capas
# convolucionales a modo de ejemplo que luego la librería descartará al crear el modelo ya que el tipo de entrada no se
# corresponde con el necesario para realizar convoluciones. Para que se realizasen tendriamos que marcar el parámetro
# img_input=False al construir el problema más adelante).
net_architecture = params.actor_critic_net_architecture(
                    actor_dense_layers=2,                           critic_dense_layers=2,
                    actor_n_neurons=[256, 256],                     critic_n_neurons=[256, 256],
                    actor_dense_activation=['relu', 'relu'],        critic_dense_activation=['relu', 'relu']
                    )

# Descomentar para ejecutar el ejemplo discreto
problem = rl_problem.Problem(environment, agent, model_params, net_architecture=net_architecture, n_stack=10)

# En este caso se utiliza el parámetro max_step_epi=500 para indicar que cada episodio termine a las 500 épocas o
# iteraciones ya que por defecto este entorno llega hasta 1000. Esto es util para entornos que no tengan definido un
# máximo de épocas.
problem.solve(40, render=True, max_step_epi=500, render_after=90, skip_states=5)
problem.test(render=True, n_iter=10)