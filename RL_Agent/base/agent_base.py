import numpy as np
from abc import ABCMeta, abstractmethod


class AgentInterface(object, metaclass=ABCMeta):
    """
    This class is an interface for building reinforcement learning agents. Here are the definitions of the methods that
    are required for an agent to work in the library.
    """
    def __init__(self):
        self.state_size = None  # (tuple) size and shape of states.
        self.n_actions = None  # (int) Number of actions.
        self.stack = None  # (bool) True means that a sequence of input in contiguous time steps are stacked in the state.
        self.img_input = None  # (bool) True if states are images (3D array), False if states are 1D array.
        self.agent_name = None   # (str) id of the agent.

    @abstractmethod
    def build_agent(self):
        """
        Define the agent params, structure, architecture, neural nets ...
        """
        pass

    def compile(self):
        pass

    @abstractmethod
    def act_train(self, obs):
        """
        Select an action given an observation :param obs: (numpy nd array) observation or state.
        :return: (int or [floats]) int if actions are discrete or numpy array of float of action shape if actions are
            continuous)
        """
        pass

    @abstractmethod
    def act(self, obs):
        """
        Select an action given an observation in only exploitation mode.
        :return: (int or [floats]) int if actions are discrete or numpy array of float of action shape if actions are
            continuous)
        """
        pass

    @abstractmethod
    def remember(self, obs, action, reward, next_obs, done):
        """
        Store an experience in memory for training the agent.
        :param obs: (numpy nd array). Current Observation (State), numpy array with state shape.
        :param action: (int, [int] or [floats]) Action selected, numpy array of actions. If actions are discrete an
            unique int can be used or a hot encoded array of ints. If actions are continuous an array of float should
            be used.
        :param reward: (float). Reward for the action taken in the current state.
        :param next_obs:  (numpy nd array). Next Observation (Next State), numpy arrays with state shape.
        :param done: (bool). Flag for episode finished. True if next_obs is a final state.
        """
        pass

    @abstractmethod
    def replay(self):
        """
        Run the train step for the agent.
        """
        pass

    @abstractmethod
    def _load(self, path):
        """
        Load a tensorflow or keras model.
        :param path: (str) file name
        """
        pass

    @abstractmethod
    def _save_network(self, path):
        """
        Save a tensorflow or keras model.
        :param path: (str) file name
        """
        pass

    def copy_model_to_target(self):
        """
        Copy the main neural network model to a target model for stabilizing the training process.
        This is not an abstract method because may be not needed.
        """
        pass


class AgentSuper(AgentInterface):
    """
    All agents in this library inherit from this class. Here can be found basic and useful utilities for agents
    implementation.
    """
    def __init__(self, learning_rate=None, actor_lr=None, critic_lr=None, batch_size=None, epsilon=None,
                 epsilon_decay=None, epsilon_min=None, gamma=None, tau=None, n_step_return=None, memory_size=None,
                 loss_clipping=None, loss_critic_discount=None, loss_entropy_beta=None, lmbda=None, train_steps=None,
                 exploration_noise=None, n_stack=None, img_input=None, state_size=None, n_parallel_envs=None,
                 save_base_dir=None, save_model_name=None, save_each_n_iter=None,
                 net_architecture=None):

        """
        Abstract agent class for defining the principal attributes of an rl agent.
        :param learning_rate: (float) learning rate for training the agent NN. Not used if actor_lr or critic_lr are 
            defined.
        :param actor_lr: (float) learning rate for training the actor NN of an Actor-Critic agent.
        :param critic_lr: (float) learning rate for training the critic NN of an Actor-Critic agent.
        :param batch_size: (int) batch size for training procedure.
        :param epsilon: (float) exploration-exploitation rate during training. epsilon=1.0 -> Exploration,
            epsilon=0.0 -> Exploitation.
        :param epsilon_decay: (float) exploration-exploitation rate reduction factor. Reduce epsilon by multiplication
            (new epsilon = epsilon*epsilon_decay)
        :param epsilon_min: (float) min exploration-exploitation rate allowed during training.
        :param gamma: (float) Discount or confidence factor for target value estimation.
        :param tau: (float) Transference factor between main and target discriminator.
        :param n_step_return: (int) Number of steps used for calculating the return.
        :param memory_size: (int) Size of experiences memory.
        :param loss_clipping: (float) Loss clipping factor for PPO.
        :param loss_critic_discount: (float) Discount factor for critic loss of PPO.
        :param loss_entropy_beta: (float) Discount factor for entropy loss of PPO.
        :param lmbda: (float) PPO lambda factor.
        :param train_steps: (int) Train steps for each training iteration.
        :param exploration_noise: (float) Standard deviation of a normal distribution for selecting actions during PPO
            training.
        :param n_stack: (int) Number of time steps stacked on the state (observation stacked).
        :param img_input: (bool) Flag for using a images as states. True state are images (3D array).
        :param state_size: (tuple of ints) State size. Needed if the original environment state size is modified by any
            preprocessing.
        :param n_parallel_envs: (int) Number of parallel environments when using A3C or PPO. By default number of cpu
            kernels are selected.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            RL_Agent.base.utils.networks
        """
        super().__init__()

        self.learning_rate = learning_rate
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.batch_size = batch_size

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.gamma = gamma
        self.tau = tau

        self.memory_size = memory_size
        self.loss_clipping = loss_clipping
        self.critic_discount = loss_critic_discount
        self.entropy_beta = loss_entropy_beta
        self.lmbda = lmbda
        self.train_epochs = train_steps
        self.exploration_noise = exploration_noise

        self.n_step_return = n_step_return

        self.n_stack = n_stack
        self.img_input = img_input
        self.state_size = state_size
        self.env_state_size = state_size

        self.n_parallel_envs = n_parallel_envs

        self.net_architecture = net_architecture

        self.save_if_better = True

        self.optimizer = None
        self.agent_builded = False

        self.save_base = save_base_dir
        self.save_name = save_model_name
        self.save_each = save_each_n_iter

    def build_agent(self, state_size, n_actions, stack):
        """
        Define the agent params, structure, architecture, neural nets ...
        :param state_size: (tuple of ints) State size.
        :param n_actions: (int) Number of actions.
        :param stack: (bool) True means that a sequence of input in contiguous time steps are stacked in the state.
        """
        super().build_agent()

        self.state_size = state_size
        self.n_actions = n_actions
        self.stack = stack
        self.agent_builded = True

    def compile(self):
        super().compile()

    def _format_obs_act(self, obs):
        """
        Reshape the observation (state) to fits the neural network inputs.
        :param obs: (nd array) Observation (state) array of state shape.
        :return: (nd array)
        """
        if self.img_input:
            if self.stack:
                obs = np.dstack(obs)
            obs = np.array([obs])

        elif self.stack:
            obs = np.array([obs])
        else:
            obs = np.array(obs).reshape(-1, self.state_size)

        return obs

    def set_batch_size(self, batch_size):
        """
        Method for change the batch size for the neural net training.
        :param batch_size: (int).
        """
        self.batch_size = batch_size

    def set_gamma(self, gamma):
        """
        Method for change the discount or confidence factor for target state value.
        :param gamma: (float).
        """
        self.gamma = gamma

    def set_train_steps(self, train_epochs):
        """
        Method for change the number of train steps for the neural net in each training execution of the neural net.
        :param train_epochs: (int).
        """
        self.train_epochs = train_epochs

    def set_optimizer(self, opt):
        """
        Method for change optimizer used for the neural net training.
        :param opt: (keras optimizer or keras optimizer id).
        """
        self.optimizer = opt