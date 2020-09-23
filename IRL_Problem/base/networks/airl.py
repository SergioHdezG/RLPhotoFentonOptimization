import tensorflow as tf
import numpy as np
import os.path as path

# Network for the Actor Critic
class ACNet(object):
    def __init__(self, scope, sess, state_size, n_actions, stack=False, img_input=False, lr_actor=0.0001,
                 lr_critic=0.001):
        self.state_size = state_size
        self.n_actions = n_actions
        pass

    def _build_graph(self):
        self.s_t = tf.placeholder(tf.float32, [None, self.state_size], "obs")  # states tensor
        self.ns_t = tf.placeholder(tf.float32, [None, self.state_size], "obs")  # next states tensor
        self.a_t = tf.placeholder(tf.float32, [None, self.n_actions], "actions")  # actions tensor

        with tf.variable_scope("net"):
            self.reward, self.v, self.v_nrext = self._build_net("net")

            net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="net")

            with tf.name_scope('local_grad'):
                self.a_grads = tf.gradients(self.a_loss,
                                            self.a_params)  # calculate gradients for the network weights
                self.c_grads = tf.gradients(self.c_loss, self.c_params)

        with tf.name_scope('sync'):  # update local and global network weights
            with tf.name_scope('push'):
                self.update_a_op = self.actor_optimizer.apply_gradients(zip(self.a_grads, self.a_params), name="update_a_op")
                self.update_c_op = self.critic_optimizer.apply_gradients(zip(self.c_grads, self.c_params), name="update_c_op")

    def _build_net(self, scope):  # neural network structure of the actor and critic
        with tf.variable_scope('reward'):
            input = tf.concat([self.s_t, self.a_t], axis=1)
            r_dens1 = tf.keras.layers.Dense(32, input_shape=self.state_size, activation='relu', name='la')(input)
            r_dens2 = tf.keras.layers.Dense(32, input_shape=self.state_size, activation='relu', name='la')(r_dens1)
            r_out = tf.keras.layers.Dense(1, activation='tanh', name='la')(r_dens2)

        with tf.variable_scope('value'):
            v_dens1 = tf.keras.layers.Dense(32, input_shape=self.state_size, activation='relu', name='la')(self.s_t)
            v_dens2 = tf.keras.layers.Dense(32, input_shape=self.state_size, activation='relu', name='la')(v_dens1)
            v_out = tf.keras.layers.Dense(1, activation='tanh', name='la')(v_dens2)

        with tf.variable_scope('next_value'):
            nv_dens1 = tf.keras.layers.Dense(32, input_shape=self.state_size, activation='relu', name='la')(self.ns_t)
            nv_dens2 = tf.keras.layers.Dense(32, input_shape=self.state_size, activation='relu', name='la')(nv_dens1)
            nv_out = tf.keras.layers.Dense(1, activation='tanh', name='la')(nv_dens2)

        return r_out, v_out, nv_out


    def load(self, dir, name):
        name = path.join(dir, name)
        loaded_model = tf.train.import_meta_graph(name)
        loaded_model.restore(self.sess, tf.train.latest_checkpoint(dir+"./"))
