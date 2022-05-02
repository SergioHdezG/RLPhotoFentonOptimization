class EnvInterfaz(object):
    def __init__(self):
        self.action_space = ActionSpaceInterfaz()
        self.observation_space = None


    def reset(self):
        """
        :return: observation. numpy array of state shape
        """
        # return obs
        pass

    def step(self, action):
        """
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

class ActionSpaceInterfaz:
    def __init__(self):
        self.n = 0  # Number of actions
        self.action_space = None  # List of actions
