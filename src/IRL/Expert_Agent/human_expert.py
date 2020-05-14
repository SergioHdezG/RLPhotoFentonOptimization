from collections import deque

import gym
import gym.utils.play as gymplay
import numpy as np
from src.IRL.utils.callbacks import Callbacks

class HumanExpert:
    def __init__(self, environment, n_stack, img_input, state_size):
        # Inicializar el entorno
        if isinstance(environment, str):
            try:
                self.env = gym.make(environment)
                self.is_gym_env = True
            except:
                print(environment, "is not listed in gym environmets")
        else:
            try:
                self.env = environment.env()
                self.is_gym_env = False
            except:
                print("The constructor of your environment is not well defined. "
                      "To use your own environment you need a constructor like: env()")

        self.n_stack = n_stack
        self.img_input = img_input
        self.state_size = state_size

        # # Stacking inputs
        # if self.n_stack is not None and self.n_stack > 1:
        #     self.obs_queue = deque(maxlen=self.n_stack)
        #     self.obs_next_queue = deque(maxlen=self.n_stack)
        # else:
        #     self.obs_queue = None
        #     self.obs_next_queue = None

    def play(self, render=True, n_iter=500):

        # dict = {
        #     (ord("s"),): 0,
        #     (ord("a"),): 1,
        #     (ord("w"),): 2,
        #     (ord("d"),): 3,
        #     (32, 100): 4,
        #     (32, 97): 5
        # }
        callback = Callbacks()
        if self.is_gym_env:
            gymplay.play(self.env, zoom=1, callback=callback.remember_callback)  #, keys_to_action=dict)

        return callback.memory

