# Photo-fenton model optimization through Deep Reinforcement Learning

Code for optimizing the parametric Photo-fenton model from [[1]](#1) using deep reinforcement learning.
Here we include some different configuration over a Proximal Policy Optimization (PPO) agent [[1]](#1) including 
balancing the memory of experiences, Hindsight Experience Replay [[3]](#1) and including expert knowledge.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SergioHdezG/RLPhotoFentonOptimization/HEAD)
To run an experiment:
```bash
python <file.py> <path to folder containing exp_config.yaml>
```

E. g.: To run an optimization of peroxide model using a PPO agent with "balanced" memory of experiences and "random" 
initialization, configure the exp_config.yaml file:

```bash
# Optimization conf
actor_lr: 1e-5
critic_lr: 1e-5
batch_size: 128
exploration_noise: 5.0
epsilon: 1.0
epsilon_decay: 0.8
epsilon_min: 0.09  # exploration_noise*epsilon_min > 0.4
memory_size: 40
histogram_memory: False
n_stack: 20
n_step_return: 20
skip_states: 1
iter: 25

sodis_params: src/environments/fotocaos_complete_model/sodis_params.txt
perox_params: src/environments/fotocaos_complete_model/perox_params.txt

experiment_path: experimentos/

test_iter: 0

test: False

optimize_peroxide: True
optimize_bacteria: False
```

Then, run the .py file and pass the path to the folder containing the exp_config.yaml file:

```bash
python Agent_DB-R.py ./
```

## Installation

Install python 3.6 and then install the requirements.txt

```bash
pip install -r requirements.txt
```

## References
<a id="1">[1]</a> 
C. Casado, J. Moreno-SanSegundo, I. De la Obra, B. Esteban
Garc ́ıa, J. A. S ́anchez P ́erez, and J. Marug ́an, “Mechanistic mod-
elling of wastewater disinfection by the photo-fenton process
at circumneutral ph,” Chemical Engineering Journal, vol. 403, p.
126335, 2021.
https://doi.org/10.1016/j.cej.2020.126335

<a id="1">[2]</a> 
J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov
“Proximal policy optimization algorithms,” arXiv preprint
arXiv:1707.06347, 2017. https://arxiv.org/abs/1707.06347

<a id="1">[3]</a> 
M. Andrychowicz, F. Wolski, A. Ray, J. Schneider, R. Fong,
P. Welinder, B. McGrew, J. Tobin, O. Pieter Abbeel, and
W. Zaremba, “Hindsight experience replay,” in Advances in Neu-
ral Information Processing Systems, ser. NeurIPS, I. Guyon, U. V.
Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and
R. Garnett, Eds., vol. 30, 2017.
