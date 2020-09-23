import random

import gym
import multiprocessing
import threading
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf

# PARAMETERS
OUTPUT_GRAPH = True  # safe logs
RENDER = True  # render one worker
LOG_DIR = './log'  # savelocation for logs
N_WORKERS = multiprocessing.cpu_count()  # number of workers
MAX_EP_STEP = 500  # maxumum number of steps per episode
MAX_GLOBAL_EP = 5000  # total number of episodes
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10  # sets how often the global net is updated
GAMMA = 0.90  # discount factor
ENTROPY_BETA = 0.01  # entropy factor
LR_A = 0.0001  # learning rate for actor
LR_C = 0.001  # learning rate for critic

# set environment
GAME = 'CartPole-v1'
env = gym.make(GAME)
env.reset()
# if RENDER:                 # uncomment if rendering does not work
#    env.render()
N_S = env.observation_space.shape[0]  # number of states
N_A = env.action_space.n  # number of actions
# A_BOUND = [env.action_space.low, env.action_space.high]  # action bounds


# Network for the Actor Critic
class ACNet(object):
    def __init__(self, scope, sess, globalAC=None):
        self.sess = sess
        self.actor_optimizer = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')  # optimizer for the actor
        self.critic_optimizer = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')  # optimizer for the critic

        if scope == GLOBAL_NET_SCOPE:  # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')  # state
                self.a_params, self.c_params = self._build_net(scope)[-2:]  # parameters of actor and critic net
        else:  # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')  # state
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')  # action
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')  # v_target value

                p_a, self.v, self.a_params, self.c_params = self._build_net(
                    scope)  # get mu and sigma of estimated action from neural net

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                normal_dist = (p_a * self.a_his) + 1e-10  # +1e-10 to prevent zero values

                with tf.name_scope('a_loss'):
                    log_prob = tf.log(normal_dist)
                    exp_v = log_prob * td
                    entropy = -(p_a*tf.log(p_a + 1e-10))  # normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = p_a
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss,
                                                self.a_params)  # calculate gradients for the network weights
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):  # update local and global network weights
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = self.actor_optimizer.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = self.critic_optimizer.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):  # neural network structure of the actor and critic
        with tf.variable_scope('actor'):
            l_a = tf.keras.layers.Dense(200, activation='relu', name='la')(self.s)
            p_a = tf.keras.layers.Dense(N_A, activation='softmax', name='mu')(l_a)  # estimated action value
        with tf.variable_scope('critic'):
            l_c = tf.keras.layers.Dense(100, activation='relu', name='lc')(self.s)
            v = tf.keras.layers.Dense(1, name='v')(l_c)  # estimated value for state
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return p_a, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        p_a = self.sess.run(self.A, {self.s: s})[0]
        return np.random.choice(N_A, p=p_a)


# worker class that inits own environment, trains on it and updloads weights to global net
class Worker(object):
    def __init__(self, name, globalAC, sess):
        self.env = gym.make(GAME) #.unwrapped  # make environment for each worker
        self.name = name
        self.AC = ACNet(name, sess, globalAC)  # create ACNet for each worker
        self.sess = sess
        self.epsilon = 0.4

    def work(self):
        global global_rewards, global_episodes
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not coord.should_stop() and global_episodes < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            for ep_t in range(MAX_EP_STEP):
                if self.name == 'W_0' and RENDER:
                    self.env.render()
                a = self.act(s) #self.AC.choose_action(s)  # estimate stochastic action based on policy
                s_, r, done, info = self.env.step(a)  # make step in environment
                if not done:
                    done = True if ep_t == MAX_EP_STEP - 1 else False

                ep_r += r
                # save actions, states and rewards in buffer
                buffer_s.append(s)
                act_one_hot = np.zeros(N_A)  # turn action into one-hot representation
                act_one_hot[a] = 1
                buffer_a.append(act_one_hot)
                buffer_r.append((r + 8) / 8)  # normalize reward

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = self.sess.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(
                        buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)  # actual training step, update global ACNet
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()  # get global parameters to local ACNet

                s = s_
                total_step += 1
                if done:
                    if len(global_rewards) < 5:  # record running episode reward
                        global_rewards.append(ep_r)
                    else:
                        global_rewards.append(ep_r)
                        global_rewards[-1] = (np.mean(global_rewards[-5:]))  # smoothing
                    print(
                        self.name,
                        "Ep:", global_episodes,
                        "| Ep_r: %i" % global_rewards[-1],
                    )
                    global_episodes += 1
                    break

    def act(self, obs):
        """
        Selecting the action using epsilon greedy policy
        :param obs: Observation (State)
        :return:
        """
        if np.random.rand() <= self.epsilon:
            self.epsilon *= 0.99995
            return random.randrange(N_A)  # '''<-- Exploration'''

        # if self.stack:
        #     if self.img_input:
        #         obs = obs.reshape(-1, *self.state_size)
        #     else:
        #         obs = obs.reshape(-1, *self.state_size)
        # else:
        #     if self.img_input:
        #         obs = obs.reshape(-1, *self.state_size)
        #     else:
        # obs = obs.reshape(-1, self.state_size)
        return self.AC.choose_action(obs)  # '''<-- Exploitation'''


if __name__ == "__main__":
    global_rewards = []
    global_episodes = 0

    sess = tf.Session()

    with tf.device("/cpu:0"):
        global_ac = ACNet(GLOBAL_NET_SCOPE, sess)  # we only need its params
        workers = []
        # Create workers
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i  # worker name
            workers.append(Worker(i_name, global_ac, sess))

    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:  # write log file
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, sess.graph)

    worker_threads = []
    for worker in workers:  # start workers
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)  # wait for termination of workers

    plt.plot(np.arange(len(global_rewards)), global_rewards)  # plot rewards
    plt.xlabel('step')
    plt.ylabel('total moving reward')
    plt.show()


