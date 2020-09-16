import numpy as np
from CAPORL.RL_Agent.PPO.ppo_agent_base import PPOSuper

def create_agent():
    return "PPO_discrete_async"

# worker class that inits own environment, trains on it and updloads weights to global net
class Agent(PPOSuper):
    def __init__(self, state_size, n_actions, stack=False, img_input=False, lr_actor=0.0001, lr_critic=0.001,
                 batch_size=32, buffer_size=2048, epsilon=1.0, epsilon_decay=0.995, epsilon_min = 0.1,
                 net_architecture=None, n_asyn_envs=2):
        super().__init__(state_size, n_actions, stack=stack, img_input=img_input, lr_actor=lr_actor,
                        lr_critic=lr_critic, batch_size=batch_size, buffer_size=buffer_size, epsilon=epsilon,
                        epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, net_architecture=net_architecture)

        self.loss_selected = self.proximal_policy_optimization_loss_discrete
        self.actor, self.critic = self._build_model(net_architecture, last_activation='softmax')
        self.n_asyn_envs = n_asyn_envs
        self.dummy_action, self.dummy_value = self.dummies_parallel(self.n_asyn_envs)
        self.remember = self.remember_parallel

    def act(self, obs):
        obs = self._format_obs_act_parall(obs)
        p = self.actor.predict([obs, self.dummy_value, self.dummy_action, self.dummy_value, self.dummy_value])

        action = [np.random.choice(self.n_actions, p=np.nan_to_num(p[i])) for i in range(self.n_asyn_envs)]
        value = self.critic.predict([obs])
        action_matrix = [np.zeros(self.n_actions) for i in range(self.n_asyn_envs)]
        for i in range(self.n_asyn_envs):
            action_matrix[i][action[i]] = 1
        return action, action_matrix, p, value

    def act_test(self, obs):
        obs = self._format_obs_act(obs)

        p = self.actor.predict([obs, self.dummy_value, self.dummy_action, self.dummy_value, self.dummy_value])
        action = np.argmax(p[0])
        return action