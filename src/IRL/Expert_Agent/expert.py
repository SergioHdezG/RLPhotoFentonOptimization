import gym
from src.IRL.Expert_Agent.human_expert import HumanExpert
from src.IRL.Expert_Agent.rl_expert import RLExpert
class Expert:
    def __init__(self, environment, expert_agent_type, n_stack, img_input, state_size):

        self.n_stack = n_stack
        self.img_input = img_input

        # Set state_size depending on the input type
        if state_size is None:
            if img_input:
                self.state_size = self.env.observation_space.shape
            else:
                self.state_size = self.env.observation_space.shape[0]
        else:
            self.state_size = state_size

        if expert_agent_type == "human_expert":
            self.expert_agent = HumanExpert(environment, self.n_stack, self.img_input, self.state_size)
        else:
            self.expert_agent = RLExpert(environment, expert_agent_type, self.n_stack, self.img_input, self.state_size)


    def play(self):
        return self.expert_agent.play(render=False, n_iter=500)

    def trajectories(self):
        return self.expert_agent.callback.memory
