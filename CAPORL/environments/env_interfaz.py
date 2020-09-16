class EnvInterface(object):
    """
    This class is an interface for building custom environments. It is based on gym (from OpenAI) environment interfaces
    but using this interface you can avoid create a custom gym environment.
    """
    def __init__(self):
        self.action_space = ActionSpaceInterface()
        self.observation_space = None

    def reset(self):
        """
        Reset the environment to an initial state
        :return: observation. numpy array of state shape
        """
        # return obs
        pass

    def step(self, action):
        """
        Take an action and executes it
        :param action:
        :return: observation (numpy array), reward (float), done (bool), info (dict or None)
        """
        # return state, reward, done, None
        pass

    def render(self):
        """
        Render the environment.
        """
        pass

    def close(self):
        """
        Close rendering window. This may not be needed.
        """
        pass

class ActionSpaceInterface(object):
    """
    This class defines the ActionSpaceInterface type used in EnvInterface.
    """
    def __init__(self):
        self.n = 0  # Number of actions
        self.action_space = None  # List of actions
