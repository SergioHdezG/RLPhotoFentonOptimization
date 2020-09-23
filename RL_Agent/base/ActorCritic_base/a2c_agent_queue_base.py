import numpy as np
from RL_Agent.base.utils.Memory.deque_memory import Memory
from RL_Agent.base.ActorCritic_base.a2c_agent_base import A2CSuper

# worker class that inits own environment, trains on it and updloads weights to global net
class A2CQueueSuper(A2CSuper):
    def __init__(self, sess, state_size, n_actions, stack=False, img_input=False, lr_actor=0.0001, lr_critic=0.001,
                 n_steps_update=10, net_architecture=None, continuous_actions=False, batch_size=32):
        super().__init__(sess, state_size, n_actions, stack=stack, img_input=img_input, lr_actor=lr_actor,
                         lr_critic=lr_critic, n_steps_update=n_steps_update, net_architecture=net_architecture,
                         continuous_actions=continuous_actions)
        self.batch_size = batch_size

        self.memory = Memory(maxlen=10000)
        self.episode_memory = []

    def remember_episode(self, obs, action, v_target):
        """
        Store a memory in a list of memories
        :param obs: Current Observation (State)
        :param action: Action selected
        :param reward: Reward
        :param next_obs: Next Observation (Next State)
        :param done: If the episode is finished
        :return:
        """
        for o, a, v in zip(obs, action, v_target):
            self.memory.append([o, a, v])

    def load_main_memories(self):
        """
        Load a batch of memories
        :return: Current Observation, Action, Reward, Next Observation, episode finished flag
        """
        _, memory, _ = self.memory.sample(self.batch_size)
        # memory = np.array(random.sample(self.memory, self.batch_size))
        obs, action, v_target = memory[:, 0], memory[:, 1], memory[:, 2]
        obs = np.array([x.reshape(self.state_size) for x in obs])
        action = np.array([x.reshape(self.n_actions) for x in action])
        v_target = np.vstack(v_target)
        return obs, action, v_target

    def load_episode_memories(self):
        """
        Load a batch of memories
        :return: Current Observation, Action, Reward, Next Observation, episode finished flag
        """
        episode_memory = np.array(self.episode_memory)
        obs, action, reward = episode_memory[:, 0], episode_memory[:, 1], episode_memory[:, 2]
        # obs = np.array([x.reshape(self.state_size) for x in obs])
        # action = np.array([x.reshape(self.n_actions) for x in action])
        self.episode_memory = []
        return obs, action, reward

    def _replay(self):
        """"
        Training process
        """
        if self.memory.len() > self.batch_size:
            obs_buff, actions_buff, buffer_v_target = self.load_main_memories()
            feed_dict = {
                self.worker.s: obs_buff,
                self.worker.a_his: actions_buff,
                self.worker.v_target: buffer_v_target,
            }
            self.worker.update_global(feed_dict)  # actual training step, update global ACNet

        if self.total_step % self.n_steps_update == 0 or self.done:  # update global and assign to local net
            obs_buff, actions_buff, reward_buff = self.load_episode_memories()

            if self.done:
                v_next_obs = 0  # terminal
            else:
                v_next_obs = self.sess.run(self.worker.v, {self.worker.s: self.next_obs[np.newaxis, :]})[0, 0]
            buffer_v_target = []

            for r in reward_buff[::-1]:  # reverse buffer r
                v_next_obs = r + self.gamma * v_next_obs
                buffer_v_target.append(v_next_obs)
            buffer_v_target.reverse()

            self.remember_episode(obs_buff, actions_buff, buffer_v_target)

        self.total_step += 1

    def set_memory(self, memory, maxlen):
        self.memory = memory(maxlen=maxlen)