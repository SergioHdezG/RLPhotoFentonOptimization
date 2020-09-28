import numpy as np
from RL_Agent.base.PPO_base.ppo_agent_base import PPOSuper
from RL_Agent.base.utils import agent_globals


def create_agent():
    return agent_globals.names["ppo_discrete"]

# worker class that inits own environment, trains on it and updloads weights to global net
class Agent(PPOSuper):
    def __init__(self):
        self.agent_name = agent_globals.names["ppo_discrete"]

    def build_agent(self, state_size, n_actions, stack=False, img_input=False, lr_actor=0.0001, lr_critic=0.001,
                 batch_size=32, buffer_size=2048, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1,
                 net_architecture=None):
        super().__init__(state_size, n_actions, stack=stack, img_input=img_input, lr_actor=lr_actor,
                         lr_critic=lr_critic, batch_size=batch_size, buffer_size=buffer_size, epsilon=epsilon,
                         epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, net_architecture=net_architecture)

        self.agent_name = agent_globals.names["ppo_discrete"]
        self.loss_selected = self.proximal_policy_optimization_loss_discrete
        self.actor, self.critic = self._build_model(net_architecture, last_activation='softmax')
        self.dummy_action, self.dummy_value = self.dummies_sequential()

    def act(self, obs):
        obs = self._format_obs_act(obs)

        p = self.actor.predict([obs, self.dummy_value, self.dummy_action, self.dummy_value, self.dummy_value])
        action = np.random.choice(self.n_actions, p=np.nan_to_num(p[0]))

        value = self.critic.predict([obs])[0]
        action_matrix = np.zeros(self.n_actions)
        action_matrix[action] = 1
        return action, action_matrix, p, value

    def act_test(self, obs):
        obs = self._format_obs_act(obs)
        p = self.actor.predict([obs, self.dummy_value, self.dummy_action, self.dummy_value, self.dummy_value])
        action = np.argmax(p[0])
        return action
