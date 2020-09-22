from CAPORL.RL_Problem.rl_problem_super import RLProblemSuper


def create_agent():
    return 'DDPG'


class DDPGPRoblem(RLProblemSuper):
    """
    Deep Deterministic Policy Gradient Problem extend RLProblemSuper
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
        # self.env = self.env.unwrapped
        # self.env.seed(1)

        # Action bouds, only for cotinuous action spaces.
        self.action_low_bound = self.env.action_space.low
        self.action_high_bound = self.env.action_space.high

        self._build_agent(agent, model_params, net_architecture)


    def _build_agent(self, agent, model_params, net_architecture):
        # Building the agent depending of the input type
        if self.img_input:
            stack = self.n_stack is not None and self.n_stack > 1
            state_size = (*self.state_size[:2], self.state_size[-1] * self.n_stack)
            # if self.n_stack is not None and self.n_stack > 1:
            #     return agent.Agent(self.n_actions, (*self.state_size[:2], self.state_size[-1] * self.n_stack),
            #                        self.action_low_bound, self.action_high_bound,
            #                        stack=True, img_input=self.img_input, model_params=model_params,
            #                        net_architecture=net_architecture)
            # else:
            #     return agent.Agent(self.n_actions, (*self.state_size[:2], self.state_size[-1]),
            #                        self.action_low_bound, self.action_high_bound,
            #                        img_input=self.img_input, model_params=model_params,
            #                        net_architecture=net_architecture)
        elif self.n_stack is not None and self.n_stack > 1:
            stack = True
            state_size = (self.n_stack, self.state_size)
            # return agent.Agent(self.n_actions, (self.n_stack, self.state_size), self.action_low_bound, self.action_high_bound,
            #             stack=True, model_params=model_params, net_architecture=net_architecture)
        else:
            stack = False
            state_size = self.state_size
            # return agent.Agent(self.n_actions, self.state_size, self.action_low_bound, self.action_high_bound,
            #                    model_params=model_params, net_architecture=net_architecture)
        agent.build_agent(self.n_actions, state_size, self.action_low_bound, self.action_high_bound, stack=stack,
                    img_input=self.img_input, model_params=model_params, net_architecture=net_architecture)

    def _max_steps(self, done, epochs, max_steps):
        if not done and max_steps is not None:
            return epochs >= max_steps
        return done
