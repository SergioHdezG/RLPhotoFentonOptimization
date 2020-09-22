import tensorflow as tf

from CAPORL.utils.parse_utils import *

from CAPORL.RL_Problem.rl_problem_super import *


class A2CProblem(RLProblemSuper):
    """
    Asynchronous Advantage Actor-Critic.
    This algorithm is the only one whitch does not extend RLProblemSuper because it has a different architecture.
    """
    def __init__(self, environment, agent, n_stack=1, img_input=False, state_size=None, model_params=None,
                 saving_model_params=None, net_architecture=None):
        """
        Attributes:
                environment:    Environment selected for this problem
                agent:          Agent to solve the problem: DDPG.
                n_stack:        Int >= 1. If 1, there is no stacked input. Number of time related input stacked.
                img_input:      Bool. If True, input data is an image.
                state_size:     None, Int or Tuple. State dimensions. If None it will be calculated automatically. Int
                                or Tuple format will be useful when preprocessing change the input dimensions.
                model_params:   Dictionary of params like learning rate, batch size, epsilon values, n step returns...
        """
        super().__init__(environment, agent, n_stack=n_stack, img_input=img_input, state_size=state_size,
                         saving_model_params=saving_model_params, net_architecture=net_architecture)
        self.environment = environment

        self.env.reset()

        if model_params is not None:
            batch_size, epsilon, epsilon_min, epsilon_decay, learning_rate, n_step_rew = \
                parse_model_params(model_params)

        if "A2C_continuous" in agent.agent_name:
            self.action_bound = [self.env.action_space.low, self.env.action_space.high]  # action bounds

        self.batch_size = batch_size
        self.n_steps_update = n_step_rew
        self.lr_actor = learning_rate*0.1
        self.lr_critic = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.sess = tf.Session()

        self._build_agent(agent, model_params, net_architecture)

        self.sess.run(tf.global_variables_initializer())

    def _build_agent(self, agent, model_params, net_architecture):
        if self.img_input:
            stack = self.n_stack is not None and self.n_stack > 1
            # TODO: Tratar n_stack como ambos channel last and channel first
            state_size = (*self.state_size[:2], self.state_size[-1] * self.n_stack)

        elif self.n_stack is not None and self.n_stack > 1:
            stack = True
            state_size = (self.n_stack, self.state_size)
        else:
            stack = False
            state_size = self.state_size

        if "A2C_continuous" in agent.agent_name:
            agent.build_agent(self.sess, state_size=state_size, n_actions=self.n_actions, stack=stack,
                               img_input=self.img_input, lr_actor=self.lr_actor, lr_critic=self.lr_critic,
                               n_steps_update=self.n_steps_update, action_bound=self.action_bound,
                               batch_size=self.batch_size, net_architecture=net_architecture)
        else:
            agent.build_agent(self.sess, state_size=state_size, n_actions=self.n_actions, stack=stack,
                               img_input=self.img_input, epsilon=self.epsilon, epsilon_decay=self.epsilon_decay,
                               epsilon_min=self.epsilon_min, lr_actor=self.lr_actor, lr_critic=self.lr_critic,
                               n_steps_update=self.n_steps_update, batch_size=self.batch_size,
                               net_architecture=net_architecture)

