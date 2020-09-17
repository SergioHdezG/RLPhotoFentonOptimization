import numpy as np

class AgentInterface(object):
    """
    This class is an interface for building reinforcement learning agents. Here are the definitions of the methods that
    are required for an agent to work in the library.
    """
    def __init__(self):
        self.state_size = None
        self.n_actions = None
        self.stack = None
        self.img_input = None

    def act(self, obs):
        """
        Select an action given an observation
        :return: action as numpy array of action shape
        """
        pass

    def act_test(self, obs):
        """
        Select an action given an observation in exploitation mode
        :return: action as numpy array of action shape
        """
        pass

    def remember(self, obs, action, reward, next_obs, done):
        """
        Store an experience in a list for training the agent
        :param obs: Current Observation (State), numpy arrays of observations with state shape
        :param action: Action selected, numpy array of actions with action shape
        :param reward: numpy of rewards
        :param next_obs: Next Observation (Next State), numpy arrays of observations with state shape
        :param done: Flag for episode finished, numpy array of bools
        """
        pass

    def replay(self):
        """
        Train the agent on the experiences in memory
        """
        pass

    def load(self, dir, name):
        """
        Load a model.
        :param dir: directory
        :param name: file name without extension
        """
        pass

    def save(self, name, reward):
        """
        Save the model.
        :param name: path to save + Model name
        :param reward: Current smooth reward
        """
        pass

    def copy_model_to_target(self):
        """
        Copy the main network to a target network.
        This method may not be necessary to implement
        """
        pass

class AgentSuper(AgentInterface):
    """
    All agents in this library should inherit from this class. Here can be found basic useful utilities for agents
    implementation.
    """
    def __init__(self, state_size, n_actions, img_input, stack):
        super().__init__()

        self.state_size = state_size
        self.n_actions = n_actions
        self.stack = stack
        self.img_input = img_input

    def _format_obs_act(self, obs):
        if self.img_input:
            if self.stack:
                obs = np.dstack(obs)
            obs = np.array([obs])

        elif self.stack:
            obs = np.array([obs])
        else:
            obs = obs.reshape(-1, self.state_size)

        return obs