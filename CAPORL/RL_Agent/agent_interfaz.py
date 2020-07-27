class AgentInterfaz:
    def __init__(self):
        self.state_size = None
        self.n_actions = None
        self.stack = None
        self.img_input = None

    def act(self, obs):
        """​
        Select an action given an observation​
        :return: action as numpy array of state shape​
        """
        pass

    def act_test(self, obs):
        """​
        Select an action given an observation​ in exploitation mode
        :return: action as numpy array of state shape​
        """
        pass

    def remember(self, obs, action, reward, next_obs, done):
        """​
        Store an experience in a list for training the agent​
        :param obs: Current Observation (State), numpy array of state shape​
        :param action: Action selected, numpy array of number of actions shape​
        :param reward: float​
        :param next_obs: Next Observation (Next State), numpy array of state shape​
        :param done: If the episode is finished, bool​
        """
        pass

    def replay(self):
        """​
        Train the agent on the experiences in memory​
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
