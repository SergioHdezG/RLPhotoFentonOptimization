import numpy as np
from CAPORL.RL_Agent.PPO.ppo_agent_super import PPOSuper

def create_agent():
    return "PPO_continuous"

# worker class that inits own environment, trains on it and updloads weights to global net
class Agent(PPOSuper):
    def __init__(self, state_size, n_actions, stack=False, img_input=False, lr_actor=0.0001, lr_critic=0.001,
                 action_bound=None, batch_size=32, buffer_size=2048, epsilon=1.0, epsilon_decay=0.995, epsilon_min = 0.1,
                 net_architecture=None):
        super().__init__(state_size, n_actions, stack=stack, img_input=img_input, lr_actor=lr_actor,
                         lr_critic=lr_critic, batch_size=batch_size, buffer_size=buffer_size, epsilon=epsilon,
                         epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, net_architecture=net_architecture)

        self.action_bound = action_bound
        self.loss_selected = self.proximal_policy_optimization_loss_continuous
        self.actor, self.critic = self._build_model(net_architecture, last_activation='tanh')
        self.dummy_action, self.dummy_value = self.dummies_sequential()

    def act(self, obs):
        obs = self._format_obs_act(obs)
        p = self.actor.predict([obs, self.dummy_value, self.dummy_action, self.dummy_value, self.dummy_value])
        action = action_matrix = p[0] + np.random.normal(loc=0, scale=self.exploration_noise*self.epsilon, size=p[0].shape)
        value = self.critic.predict([obs])[0]
        return action, action_matrix, p, value

    def act_test(self, obs):
        obs = self._format_obs_act(obs)
        p = self.actor.predict([obs, self.dummy_value, self.dummy_action, self.dummy_value, self.dummy_value])
        action = p[0]
        return action