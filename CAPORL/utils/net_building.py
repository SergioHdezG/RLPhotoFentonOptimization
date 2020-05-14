from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import tensorflow as tf


def read_net_params(net_architecture, actor=False, critic=False):
    id_1 = 'conv_layers'
    id_2 = 'kernel_num'
    id_3 = 'kernel_size'
    id_4 = 'kernel_strides'
    id_5 = 'conv_activation'
    id_6 = 'dense_lay'
    id_7 = 'n_neurons'
    id_8 = 'dense_activation'
    id_9 = 'use_custom_network'
    id_10 = 'custom_network'

    if actor or critic:
        if actor:
            prefix = 'actor_'
        elif critic:
            prefix = 'critic_'

        id_1 = prefix + id_1
        id_2 = prefix + id_2
        id_3 = prefix + id_3
        id_4 = prefix + id_4
        id_5 = prefix + id_5
        id_6 = prefix + id_6
        id_7 = prefix + id_7
        id_8 = prefix + id_8
        # id_9 is unique
        id_10 = prefix + id_10

    n_conv_layers = net_architecture[id_1]
    kernel_num = net_architecture[id_2]
    kernel_size = net_architecture[id_3]
    strides = net_architecture[id_4]
    conv_activation = net_architecture[id_5]

    n_dense_layers = net_architecture[id_6]
    n_neurons = net_architecture[id_7]
    dense_activation = net_architecture[id_8]

    use_custom_net = net_architecture[id_9]
    custom_net = net_architecture[id_10]

    return n_conv_layers, kernel_num, kernel_size, strides, conv_activation, n_dense_layers, n_neurons, \
           dense_activation, use_custom_net, custom_net

def build_conv_net(net_architecture, input_shape, dddqn=False, actor=False, critic=False):

    n_conv_layers, kernel_num, kernel_size, strides, conv_activation, \
    n_dense_layers, n_neurons, dense_activation, use_custom_net, \
    custom_net = read_net_params(net_architecture, actor, critic)

    if use_custom_net:
        return custom_net(input_shape)
    else:
        model = Sequential()
        model.add(Conv2D(kernel_num[0], kernel_size=kernel_size[0], input_shape=input_shape,
                         strides=(strides[0], strides[0]), padding='same',
                         activation=conv_activation[0]))
        for i in range(1, n_conv_layers):
            model.add(Conv2D(kernel_num[i], kernel_size=kernel_size[i], strides=(strides[i], strides[i]),
                             padding='same', activation=conv_activation[i]))

        model.add(Flatten())

        if not dddqn:
            for i in range(n_dense_layers):
                model.add(Dense(n_neurons[i], activation=dense_activation[i]))
        elif dddqn:
            dense_v = Dense(n_neurons[0], activation=dense_activation[0], name="dense_valor_in")(model.output)
            dense_a = Dense(n_neurons[0], activation=dense_activation[0], name="dense_advantage_in")(model.output)
            for i in range(1, n_dense_layers):
                dense_v = Dense(n_neurons[i], activation=dense_activation[i], name="dense_valor_"+str(i))(dense_v)
                dense_a = Dense(n_neurons[i], activation=dense_activation[i], name="dense_advantage_"+str(i))(dense_a)

            return model, dense_v, dense_a

        return model

def build_stack_net(net_architecture, input_shape, dddqn=False, actor=False, critic=False):
    n_dense_layers, n_neurons, dense_activation, use_custom_net, \
    custom_net = read_net_params(net_architecture, actor, critic)[-5:]

    if use_custom_net:
        return custom_net(input_shape)
    else:
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))

        if not dddqn:
            for i in range(n_dense_layers):
                model.add(Dense(n_neurons[i], input_dim=input_shape, activation=dense_activation[i]))

        elif dddqn:
            model.add(Dense(n_neurons[0], input_dim=input_shape, activation=dense_activation[0]))
            dense_v = Dense(n_neurons[1], activation=dense_activation[1], name="dense_valor_in")(model.output)
            dense_a = Dense(n_neurons[1], activation=dense_activation[1], name="dense_advantage_in")(model.output)
            for i in range(2, n_dense_layers):
                dense_v = Dense(n_neurons[i], activation=dense_activation[i], name="dense_valor_" + str(i))(dense_v)
                dense_a = Dense(n_neurons[i], activation=dense_activation[i], name="dense_advantage_" + str(i))(dense_a)

            return model, dense_v, dense_a
        return model

def build_nn_net(net_architecture, input_shape, dddqn=False, actor=False, critic=False):
    n_dense_layers, n_neurons, dense_activation, use_custom_net, \
    custom_net = read_net_params(net_architecture, actor, critic)[-5:]

    if use_custom_net:
        return custom_net(input_shape)
    else:
        model = Sequential()
        model.add(Dense(n_neurons[0], input_dim=input_shape, activation=dense_activation[0]))

        if not dddqn:
            for i in range(1, n_dense_layers):
                model.add(Dense(n_neurons[i], activation=dense_activation[i]))
        elif dddqn:
            dense_v = Dense(n_neurons[1], activation=dense_activation[1], name="dense_valor_in")(model.output)
            dense_a = Dense(n_neurons[1], activation=dense_activation[1], name="dense_advantage_in")(model.output)
            for i in range(2, n_dense_layers):
                dense_v = Dense(n_neurons[i], activation=dense_activation[i], name="dense_valor_" + str(i))(dense_v)
                dense_a = Dense(n_neurons[i], activation=dense_activation[i], name="dense_advantage_" + str(i))(dense_a)

            return model, dense_v, dense_a
        return model

def build_ddpg_conv_critic(net_architecture, input_shape, s, a):

    n_conv_layers, kernel_num, kernel_size, strides, conv_activation, \
    n_dense_layers, n_neurons, dense_activation, use_custom_net, \
    custom_net = read_net_params(net_architecture, actor=False, critic=True)

    if use_custom_net:
        return custom_net(input_shape, s, a)
    else:
        conv_obs = Conv2D(kernel_num[0], kernel_size=kernel_size[0], input_shape=input_shape,
                         strides=(strides[0], strides[0]), padding='same',
                         activation=conv_activation[0])(s)
        for i in range(1, n_conv_layers):
            conv_obs = Conv2D(kernel_num[i], kernel_size=kernel_size[i], strides=(strides[i], strides[i]),
                             padding='same', activation=conv_activation[i])(conv_obs)

        flat = tf.keras.layers.Flatten()(conv_obs)
        bias = len(n_neurons) > 2

        lay_obs = Dense(n_neurons[0], activation=dense_activation[0], use_bias=bias)(flat)
        for i in range(1, n_dense_layers - 1):
            lay_obs = Dense(n_neurons[i], activation=dense_activation[i], use_bias=False)(lay_obs)
        lay_obs = Dense(n_neurons[-1], activation='linear', use_bias=False)(lay_obs)

        lay_act = Dense(n_neurons[0], input_dim=input_shape, activation=dense_activation[0], use_bias=bias)(a)
        for i in range(1, n_dense_layers - 1):
            lay_act = Dense(n_neurons[i], activation=dense_activation[i], use_bias=False)(lay_act)
        lay_act = Dense(n_neurons[-1], activation='linear', use_bias=False)(lay_act)

        merge = tf.keras.layers.Add()([lay_obs, lay_act])
        b_init = tf.constant_initializer(0.1)
        b = tf.get_variable(name='bias', shape=[n_neurons[-1]], initializer=b_init)
        output = tf.nn.relu(merge + b)

        return output

def build_ddpg_stack_critic(net_architecture, input_shape, s, a):
    n_dense_layers, n_neurons, dense_activation, use_custom_net, \
    custom_net = read_net_params(net_architecture, actor=False, critic=True)[-5:]

    if use_custom_net:
        return custom_net(input_shape, s, a)
    else:
        flat = tf.keras.layers.Flatten(input_shape=input_shape)(s)
        bias = len(n_neurons) > 2

        lay_obs = Dense(n_neurons[0], activation=dense_activation[0], use_bias=bias)(flat)
        for i in range(1, n_dense_layers - 1):
            lay_obs = Dense(n_neurons[i], activation=dense_activation[i], use_bias=False)(lay_obs)
        lay_obs = Dense(n_neurons[-1], activation='linear', use_bias=False)(lay_obs)

        lay_act = Dense(n_neurons[0], input_dim=input_shape, activation=dense_activation[0], use_bias=bias)(a)
        for i in range(1, n_dense_layers-1):
            lay_act = Dense(n_neurons[i], activation=dense_activation[i], use_bias=False)(lay_act)
        lay_act = Dense(n_neurons[-1], activation='linear', use_bias=False)(lay_act)

        merge = tf.keras.layers.Add()([lay_obs, lay_act])
        b_init = tf.constant_initializer(0.1)
        b = tf.get_variable(name='bias', shape=[n_neurons[-1]], initializer=b_init)
        output = tf.nn.relu(merge + b)

        return output

def build_ddpg_nn_critic(net_architecture, input_shape, s, a):
    n_dense_layers, n_neurons, dense_activation, use_custom_net, \
    custom_net = read_net_params(net_architecture, actor=False, critic=True)[-5:]

    if use_custom_net:
        return custom_net(input_shape, s, a)
    else:
        bias = len(n_neurons) > 2

        lay_obs = Dense(n_neurons[0], input_dim=input_shape, activation=dense_activation[0], use_bias=bias)(s)
        for i in range(1, n_dense_layers - 1):
            lay_obs = Dense(n_neurons[i], activation=dense_activation[i], use_bias=False)(lay_obs)
        lay_obs = Dense(n_neurons[-1], activation='linear', use_bias=False)(lay_obs)

        lay_act = Dense(n_neurons[0], input_dim=input_shape, activation=dense_activation[0], use_bias=bias)(a)
        for i in range(1, n_dense_layers-1):
            lay_act = Dense(n_neurons[i], activation=dense_activation[i], use_bias=False)(lay_act)
        lay_act = Dense(n_neurons[-1], activation='linear', use_bias=False)(lay_act)

        merge = tf.keras.layers.Add()([lay_obs, lay_act])
        b_init = tf.constant_initializer(0.1)
        b = tf.get_variable(name='bias', shape=[n_neurons[-1]], initializer=b_init)
        output = tf.nn.relu(merge + b)

    return output
