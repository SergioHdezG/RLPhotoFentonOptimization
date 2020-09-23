from RL_Problem import rl_problem
# from CAPORL.IRL.utils.clipping_reward import *
# from CAPORL.IRL.utils.preprocess import *
from src.IRL.utils.callbacks import Callbacks
class RLExpert():
    def __init__(self, environment, agent, n_stack, img_input, state_size):

        model_params = {
            "learning_rate": 1e-3,
            "batch_size": 32,
            "epsilon": 0.4,
            "epsilon_decay": 0.99995,
            "epsilon_min": 0.15,
            "n_step_return": 10,
        }

        saving_model_params = None
        self.problem = rl_problem.Problem(environment, agent, model_params, saving_model_params, n_stack=n_stack, img_input=img_input,
                                          state_size=state_size)

        # problem.preprocess = atari_assault_preprocess
        # problem.clip_norm_reward = clip_reward_atari_v2""
        self.memory = []

    def play(self, n_iter=100, render=False):
        callback = Callbacks()

        print("Training the reinforcement learning expert")
        self.problem.solve(100, render=False, max_step_epi=None)
        print("Expert obtaining trajectories")
        self.problem.test(n_iter=n_iter, render=render, callback=callback.remember_callback)

        return callback.memory
