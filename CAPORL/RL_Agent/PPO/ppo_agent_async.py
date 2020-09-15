from os import path

from tensorflow.python.keras.models import model_from_json

from  CAPORL.RL_Agent.agent_interfaz import AgentInterfaz
import numpy as np
import tensorflow as tf
from CAPORL.RL_Agent.ActorCritic.A2C_Agent.Networks import a2c_net_continuous
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from CAPORL.utils import net_building
from CAPORL.utils.networks import ppo_net
from tensorflow.keras.initializers import RandomNormal
from CAPORL.RL_Agent.agent_interfaz import AgentSuper


def create_agent():
    return "PPO_continuous_async"

# worker class that inits own environment, trains on it and updloads weights to global net
class Agent(AgentSuper):
    def __init__(self, state_size, n_actions, stack=False, img_input=False, lr_actor=0.0001, lr_critic=0.001,
                 action_bound=None, batch_size=32, buffer_size=2048, epsilon=0, epsilon_decay=0.995, epsilon_min = 0.1,
                 net_architecture=None, n_asyn_envs=2):
        super().__init__()

        self.state_size = state_size
        self.n_actions = n_actions
        self.stack = stack
        self.img_input = img_input

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = 0.99
        self.batch_size = batch_size
        self.action_bound = action_bound
        self.buffer_size = buffer_size
        self.loss_clipping = 0.2
        self.critic_discount = 0.5
        self.entropy_beta = 0.001
        self.lmbda = 0.95
        self.train_epochs = 10
        self.exploration_noise = 1.0
        self.actor, self.critic = self._build_model(net_architecture)
        # self.critic = self.build_critic()
        # self.actor = self.build_actor_continuous()
        self.memory = []
        self.epsilon = epsilon  # For epsilon-greedy
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.dummy_action, self.dummy_value = np.zeros((n_asyn_envs, self.n_actions)), np.zeros((n_asyn_envs, 1))
        self.n_asyn_envs = n_asyn_envs

    def act(self, obs):
        # if self.img_input or self.stack:
        #     # obs = obs.reshape(-1, *self.state_size)
        #     obs = obs
        # else:
        #     # obs = obs.reshape(-1, self.state_size)
        #     obs = obs

        obs = self._format_obs_act_parall(obs)
        p = self.actor.predict([obs, self.dummy_value, self.dummy_action, self.dummy_value, self.dummy_value])
        action = action_matrix = p + np.random.normal(loc=0, scale=self.exploration_noise, size=p.shape)
        value = self.critic.predict(obs)
        return action, action_matrix, p, value

    def act_test(self, obs):
        # if self.img_input:
        #     if self.stack:
        #         # obs = np.squeeze(obs, axis=3)
        #         # obs = obs.transpose(1, 2, 0)
        #         obs = np.dstack(obs)
        #     obs = np.array([obs])
        #
        # elif self.stack:
        #     obs = np.array([obs])
        # else:
        #     obs = np.array([obs])
        obs = self._format_obs_act(obs)

        p = self.actor.predict([obs, self.dummy_value, self.dummy_action, self.dummy_value, self.dummy_value])
        action = p[0]
        return action

    def remember(self, obs, action, pred_act, rewards, values, mask):
        """
        Store a memory in a list of memories
        :param obs: Current Observation (State)
        :param action: Action selected with noise
        :param pred_act: Action predicted
        :param reward: Reward
        :param next_obs: Next Observation (Next State)
        :param done: If the episode is finished
        :return:
        """
        if self.img_input:
                obs = np.transpose(obs, axes=(1, 0, 2, 3, 4))
        elif self.stack:
            obs = np.transpose(obs, axes=(1, 0, 2, 3))
        else:
            obs = np.transpose(obs, axes=(1, 0, 2))

        action = np.transpose(action, axes=(1, 0, 2))
        pred_act = np.transpose(pred_act, axes=(1, 0, 2))
        rewards = np.transpose(rewards, axes=(1, 0))
        values = np.transpose(values, axes=(1, 0, 2))
        mask = np.transpose(mask, axes=(1, 0))

        o = obs[0]
        a = action[0]
        p_a = pred_act[0]
        r = rewards[0]
        v = values[0]
        m = mask[0]

        for i in range(1, self.n_asyn_envs):
            o = np.concatenate((o, obs[i]), axis=0)
            a = np.concatenate((a, action[i]), axis=0)
            p_a = np.concatenate((p_a, pred_act[i]), axis=0)
            r = np.concatenate((r, rewards[i]), axis=0)
            v = np.concatenate((v, values[i]), axis=0)
            m = np.concatenate((m, mask[i]), axis=0)

        v = np.concatenate((v, [v[-1]]), axis=0)
        returns, advantages = self.get_advantages(v, m, r)
        advantages = np.array(advantages)
        returns = np.array(returns)

        # TODO: Decidir la solución a utilizar
        index = range(len(o))
        # index = np.random.choice(range(len(obs)), self.buffer_size, replace=False)
        self.memory = [o[index], a[index], p_a[index], returns[index], r[index], v[index],
                       m[index], advantages[index]]

    def load_memories(self):
        """
        Load a batch of memories
        :return: Current Observation, Action, Reward, Next Observation, episode finished flag
        """
        obs = self.memory[0]
        action = self.memory[1]
        pred_act = self.memory[2]
        returns = self.memory[3]
        rewards = self.memory[4]
        values = self.memory[5]
        mask = self.memory[6]
        advantages = self.memory[7]

        return obs, action, pred_act, returns, rewards, values, mask, advantages

    def replay(self):
        """"
        Training process
        """
        obs, action, old_prediction, returns, rewards, values, mask, advantages = self.load_memories()

        # pred_values = self.critic.predict(obs)

        # advantage = returns - pred_values

        # TODO: Pasar rewards o returns?
        actor_loss = self.actor.fit([obs, advantages, old_prediction, returns, values], [action], batch_size=self.batch_size, shuffle=True,
                                    epochs=self.train_epochs, verbose=False)
        critic_loss = self.critic.fit([obs], [returns], batch_size=self.batch_size, shuffle=True, epochs=self.train_epochs,
                                      verbose=False)

        return actor_loss, critic_loss

    def load(self, dir, name):
        # Create a clean graph and import the MetaGraphDef nodes.
        # new_graph = tf.Graph()
        # with tf.keras.backend.get_session() as sess:
        # Import the previously export meta graph.
        name = path.join(dir, name)
        # loaded_model = tf.train.import_meta_graph(name + '.meta')
        # # tf.keras.backend.clear_session()
        # sess = tf.keras.backend.get_session()
        # loaded_model.restore(sess, tf.train.latest_checkpoint(dir + "./"))
        # json_file = open(name+'actor'+'.json', 'r')
        # loaded_model_json = json_file.read()
        # self.actor = model_from_json(loaded_model_json)
        # json_file.close()

        # load weights into new model
        self.actor.load_weights(name+'actor'+".h5")
        # self.actor.compile(optimizer=Adam(lr=self.learning_rate))

        # json_file = open(name+'critic'+'.json', 'r')
        # loaded_model_json = json_file.read()
        # self.critic = model_from_json(loaded_model_json)
        # json_file.close()

        # load weights into new model
        self.critic.load_weights(name+'critic'+".h5")
        # self.critic.compile(optimizer=Adam(lr=self.learning_rate))
        print("Loaded model from disk")

    def save(self, name, reward):
        # sess = tf.keras.backend.get_session()  # op_input_list=(self.actor.get_layers(), self.critic.get_layers())
        # self.saver = tf.train.Saver()
        name = name + "-" + str(reward)
        # self.saver.save(sess, name)
        # serialize model to JSON
        # model_json = self.actor.to_json()
        # with open(name+'actor'+".json", "w") as json_file:
        #     json_file.write(model_json)
        # serialize weights to HDF5
        self.actor.save_weights(name+'actor'+".h5")

        # model_json = self.critic.to_json()
        # with open(name+'critic'+".json", "w") as json_file:
        #     json_file.write(model_json)
        # serialize weights to HDF5
        self.critic.save_weights(name+'critic'+".h5")
        print("Saved model to disk")

    def _build_model(self, net_architecture):
        # Neural Net for Actor-Critic Model
        if net_architecture is None:  # Standart architecture
            net_architecture = ppo_net

        # Building actor
        if self.img_input:
            actor_net = net_building.build_conv_net(net_architecture, self.state_size, actor=True)
        elif self.stack:
            actor_net = net_building.build_stack_net(net_architecture, self.state_size, actor=True)
        else:
            actor_net = net_building.build_nn_net(net_architecture, self.state_size, actor=True)

        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(self.n_actions,))
        rewards = Input(shape=(1,))
        values = Input(shape=(1,))

        actor_net.add(Dense(self.n_actions, name='output', activation='tanh', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-4, seed=None)))

        actor_model = Model(inputs=[actor_net.inputs, advantage, old_prediction, rewards, values], outputs=[actor_net.outputs])
        actor_model.compile(optimizer=Adam(lr=self.lr_actor),
                            loss=[self.proximal_policy_optimization_loss_continuous(advantage=advantage,
                                                                                    old_prediction=old_prediction,
                                                                                    rewards=rewards,
                                                                                    values=values)])
        actor_model.summary()

        # Building actor
        if self.img_input:
            critic_model = net_building.build_conv_net(net_architecture, self.state_size, critic=True)
        elif self.stack:
            critic_model = net_building.build_stack_net(net_architecture, self.state_size, critic=True)
        else:
            critic_model = net_building.build_nn_net(net_architecture, self.state_size, critic=True)

        critic_model.add(Dense(1))
        critic_model.compile(optimizer=Adam(lr=self.lr_critic), loss='mse')

        return actor_model, critic_model

    def proximal_policy_optimization_loss_continuous(self, advantage, old_prediction, rewards, values):

        def loss(y_true, y_pred):
            """
            f(x) = (1/σ√2π)exp(-(1/2σ^2)(x−μ)^2)
            X∼N(μ, σ)
            """
            var = K.square(self.exploration_noise)
            pi = 3.1415926

            # σ√2π
            denom = K.sqrt(2 * pi * var)

            # exp(-((x−μ)^2/2σ^2))
            prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
            old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))

            # exp(-((x−μ)^2/2σ^2))/(σ√2π)
            new_prob = prob_num / denom
            old_prob = old_prob_num / denom

            ratio = K.exp(K.log(new_prob + 1e-10) - K.log(old_prob + 1e-10))

            p1 = ratio * advantage
            p2 = K.clip(ratio, min_value=1 - self.loss_clipping, max_value=1 + self.loss_clipping) * advantage
            actor_loss = - K.mean(K.minimum(p1, p2))
            # TODO: Es rewards o returns
            critic_loss = self.critic_discount * K.mean(K.square(rewards - values))
            entropy = - self.entropy_beta * K.mean(-(new_prob * K.log(new_prob + 1e-10)))

            return actor_loss + critic_loss + entropy

        return loss

    def get_advantages(self, values, masks, rewards):
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.lmbda * masks[i] * gae
            returns.insert(0, gae + values[i])

        adv = np.array(returns) - values[:-1]
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

    def _format_obs_act_parall(self, obs):
        if self.img_input:
            if self.stack:
                obs = np.array([np.dstack(o) for o in obs])
            else:
                obs = obs

        elif self.stack:
            # obs = obs.reshape(-1, *self.state_size)
            obs = obs
        else:
            # obs = obs.reshape(-1, self.state_size)
            obs = obs

        return obs