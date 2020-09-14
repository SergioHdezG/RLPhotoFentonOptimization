from os import path

from tensorflow.python.keras.models import model_from_json
import random
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

def create_agent():
    return "PPO_discrete"

# worker class that inits own environment, trains on it and updloads weights to global net
class Agent(AgentInterfaz):
    def __init__(self, state_size, n_actions, stack=False, img_input=False, lr_actor=0.0001, lr_critic=0.001,
                 batch_size=32, buffer_size=2048, epsilon=1.0, epsilon_decay=0.995, epsilon_min = 0.1,
                 net_architecture=None):
        self.state_size = state_size
        self.n_actions = n_actions
        self.stack = stack
        self.img_input = img_input

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = 0.99
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.loss_clipping = 0.2
        self.critic_discount = 0.5
        self.entropy_beta = 0.001
        self.lmbda = 0.95
        self.train_epochs = 10
        self.exploration_noise = 0.3
        self.actor, self.critic = self._build_model(net_architecture)
        # self.critic = self.build_critic()
        # self.actor = self.build_actor_continuous()
        self.memory = []
        self.epsilon = epsilon  # For epsilon-greedy
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.dummy_action, self.dummy_value = np.zeros((1, self.n_actions)), np.zeros((1, 1))

    def act(self, obs):
        if self.img_input:
            if self.stack:
                # obs = np.squeeze(obs, axis=3)
                # obs = obs.transpose(1, 2, 0)
                obs = np.dstack(obs)
            obs = np.array([obs])

        elif self.stack:
            obs = np.array([obs])
        else:
            obs = obs.reshape(-1, self.state_size)

        p = self.actor.predict([obs, self.dummy_value, self.dummy_action, self.dummy_value, self.dummy_value])

        # if random.random() < self.epsilon:
        #     action = random.randrange(self.n_actions)
        #
        # else:
            # action = np.argmax(p[0])
        action = np.random.choice(self.n_actions, p=np.nan_to_num(p[0], nan=0.0))
        value = self.critic.predict([obs])[0]
        action_matrix = np.zeros(self.n_actions)
        action_matrix[action] = 1
        return action, action_matrix, p, value

    def act_test(self, obs):
        if self.img_input:
            if self.stack:
                # obs = np.squeeze(obs, axis=3)
                # obs = obs.transpose(1, 2, 0)
                obs = np.dstack(obs)
            obs = np.array([obs])

        elif self.stack:
            obs = np.array([obs])
        else:
            obs = obs.reshape(-1, self.state_size)
        p = self.actor.predict([obs, self.dummy_value, self.dummy_action, self.dummy_value, self.dummy_value])
        action = np.argmax(p[0])
        # action = np.random.choice(self.n_actions, p=np.nan_to_num(p[0], nan=0.0))
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
        values.append(values[-1])
        returns, advantages = self.get_advantages(values, mask, rewards)
        obs = np.array(obs)
        action = np.array(action)
        # pred_act = np.array(pred_act)
        # pred_act = np.reshape(pred_act, (pred_act.shape[0], pred_act.shape[2]))
        pred_act = np.array([a[0] for a in pred_act])
        # returns = np.reshape(np.array(returns), (len(returns), 1))
        # returns = np.array(returns)[:, np.newaxis]
        returns = np.array(returns)
        rewards = np.array(rewards)
        values = np.array(values)
        mask = np.array(mask)
        advantages = np.array(advantages)

        # TODO: Decidir la solución a utilizar
        index = range(len(obs))
        # index = np.random.choice(range(len(obs)), self.buffer_size, replace=False)
        self.memory = [obs[index], action[index], pred_act[index], returns[index], rewards[index], values[index],
                       mask[index], advantages[index]]

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

        actor_loss = self.actor.fit([obs, advantages, old_prediction, returns, values], [action], batch_size=self.batch_size, shuffle=True,
                                    epochs=self.train_epochs, verbose=False)
        critic_loss = self.critic.fit([obs], [returns], batch_size=self.batch_size, shuffle=True, epochs=self.train_epochs,
                                      verbose=False)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
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

        actor_net.add(Dense(self.n_actions, name='output', activation='softmax'))

        actor_model = Model(inputs=[actor_net.inputs, advantage, old_prediction, rewards, values], outputs=[actor_net.outputs])
        actor_model.compile(optimizer=Adam(lr=self.lr_actor),
                            loss=[self.proximal_policy_optimization_loss_continuous(advantage=advantage,
                                                                                    old_prediction=old_prediction,
                                                                                    returns=rewards,
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

    def proximal_policy_optimization_loss_continuous(self, advantage, old_prediction, returns, values):

        def loss(y_true, y_pred):
            new_prob = K.sum(y_true * y_pred, axis=-1)
            old_prob = K.sum(y_true * old_prediction, axis=-1)

            ratio = K.exp(K.log(new_prob + 1e-10) - K.log(old_prob + 1e-10))

            p1 = ratio * advantage
            p2 = K.clip(ratio, min_value=1 - self.loss_clipping, max_value=1 + self.loss_clipping) * advantage
            actor_loss = - K.mean(K.minimum(p1, p2))
            # TODO: Es rewards o returns
            critic_loss = self.critic_discount * K.mean(K.square(returns - values))
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