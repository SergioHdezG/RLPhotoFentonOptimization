from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import tensorflow as tf
from termcolor import colored


def read_disc_net_params(net_architecture):
    id_1 = 'state_conv_layers'
    id_2 = 'state_kernel_num'
    id_3 = 'state_kernel_size'
    id_4 = 'state_kernel_strides'
    id_5 = 'state_conv_activation'
    id_6 = 'state_dense_lay'
    id_7 = 'state_n_neurons'
    id_8 = 'state_dense_activation'

    id_9 = 'action_dense_lay'
    id_10 = 'action_n_neurons'
    id_11 = 'action_dense_activation'

    id_12 = 'common_dense_lay'
    id_13 = 'common_n_neurons'
    id_14 = 'common_dense_activation'

    id_15 = 'use_custom_network'
    id_16 = 'state_custom_network'
    id_17 = 'action_custom_network'
    id_18 = 'common_custom_network'

    id_19 = 'last_layer_activation'

    if (id_1 and id_2 and id_3 and id_4 and id_5 in net_architecture) \
            and (net_architecture[id_1] and net_architecture[id_2] and net_architecture[id_3] and net_architecture[id_4]
                 and net_architecture[id_5] is not None):

        state_n_conv_layers = net_architecture[id_1]
        state_kernel_num = net_architecture[id_2]
        state_kernel_size = net_architecture[id_3]
        state_strides = net_architecture[id_4]
        state_conv_activation = net_architecture[id_5]
        print('Convolutional layers for state net selected: {state_conv_layers:\t\t', state_n_conv_layers,
              '\n\t\t\t\t\t\t\t    state_kernel_num:\t\t\t', state_kernel_num, '\n\t\t\t\t\t\t\t    '
                                                                               'state_kernel_size:\t\t',
              state_kernel_size, '\n\t\t\t\t\t\t\t    state_kernel_strides:\t\t',
              state_strides, '\n\t\t\t\t\t\t\t    state_conv_activation:\t', state_conv_activation, '}')
    else:
        state_n_conv_layers = None
        state_kernel_num = None
        state_kernel_size = None
        state_strides = None
        state_conv_activation = None
        print(colored('WARNING: If you want to specify convolutional layers for state net you must set all the values '
                      'for the following keys: state_conv_layers, state_kernel_num, state_kernel_size, '
                      'state_kernel_strides and state_conv_activation', 'yellow'))

    if (id_6 and id_7 and id_8 in net_architecture) and (net_architecture[id_6] and net_architecture[id_7]
                                                         and net_architecture[id_8] is not None):
        state_n_dense_layers = net_architecture[id_6]
        state_n_neurons = net_architecture[id_7]
        state_dense_activation = net_architecture[id_8]
        print('Dense layers for state net selected: {state_dense_lay:\t\t\t', state_n_dense_layers, '\n\t\t\t\t\t    '
                                                                                                    'state_n_neurons:\t\t\t',
              state_n_neurons, '\n\t\t\t\t\t    dense_activation:\t', state_dense_activation,
              '}')
    else:
        state_n_dense_layers = None
        state_n_neurons = None
        state_dense_activation = None
        print(colored('WARNING: If you want to specify dense layers for state net you must set all the values for the '
                      'following keys: state_dense_lay, state_n_neurons and state_dense_activation', 'yellow'))

    if (id_9 and id_10 and id_11 in net_architecture) and (net_architecture[id_9] and net_architecture[id_10]
                                                           and net_architecture[id_11] is not None):
        action_n_dense_layers = net_architecture[id_9]
        action_n_neurons = net_architecture[id_10]
        action_dense_activation = net_architecture[id_11]
        print('Dense layers for action net selected: {action_dense_lay:\t\t\t', action_n_dense_layers, '\n\t\t\t\t\t   '
                                                                                                       'action_n_neurons:\t\t\t',
              action_n_neurons, '\n\t\t\t\t\t    action_activation:\t',
              action_dense_activation, '}')
    else:
        action_n_dense_layers = None
        action_n_neurons = None
        action_dense_activation = None
        print(colored('WARNING: If you want to specify dense layers for action net you must set all the values for the '
                      'following keys: action_dense_lay, action_n_neurons and action_dense_activation', 'yellow'))

    if (id_12 and id_13 and id_14 in net_architecture) and (net_architecture[id_12] and net_architecture[id_13]
                                                            and net_architecture[id_14] is not None):
        common_n_dense_layers = net_architecture[id_12]
        common_n_neurons = net_architecture[id_13]
        common_dense_activation = net_architecture[id_14]
        print('Dense layers for common net selected: {common_dense_lay:\t\t\t', common_n_dense_layers, '\n\t\t\t\t\t   '
                                                                                                       'common_n_neurons:\t\t\t',
              common_n_neurons, '\n\t\t\t\t\t    common_activation:\t',
              common_dense_activation, '}')
    else:
        common_n_dense_layers = None
        common_n_neurons = None
        common_dense_activation = None
        print(colored('WARNING: If you want to specify dense layers for common net you must set all the values for the '
                      'following keys: common_dense_lay, common_n_neurons and common_dense_activation', 'yellow'))

    if (id_15 and id_16 and id_17 and id_18 in net_architecture) and \
            (net_architecture[id_15] and net_architecture[id_18]
             is not None):
        use_custom_net = net_architecture[id_15]
        state_custom_net = net_architecture[id_16]
        action_custom_net = net_architecture[id_17]
        common_custom_net = net_architecture[id_18]
        print('Custom network option selected: {use_custom_network: ', use_custom_net, ', state_custom_network: ',
              state_custom_net, ', action_custom_network: ', action_custom_net, ', common_custom_network: ',
              common_custom_net, '}')
    else:
        use_custom_net = False
        state_custom_net = None
        action_custom_net = None
        common_custom_net = None
        print(colored('WARNING: If you want to use a custom neural net you must set the values at least for the '
                      'following keys: use_custom_network and common_custom_network, and additionally for: '
                      'state_custom_network and action_custom_network', 'yellow'))

    if (id_19 in net_architecture) and (net_architecture[id_19] is not None):
        last_layer_activation = net_architecture[id_19]
        print('Last layer activation: {last_layer_activation: ', last_layer_activation, '}')
    else:
        last_layer_activation = 'sigmoid'
        print(colored('WARNING: Last layer activation function was not specified, sigmoid activation is selected by '
                      'default.', 'yellow'))

    return state_n_conv_layers, state_kernel_num, state_kernel_size, state_strides, state_conv_activation, \
           state_n_dense_layers, state_n_neurons, state_dense_activation, use_custom_net, action_n_dense_layers, \
           action_n_neurons, action_dense_activation, common_n_dense_layers, common_n_neurons, \
           common_dense_activation, use_custom_net, state_custom_net, action_custom_net, common_custom_net, \
           last_layer_activation


def build_disc_nn_net(net_architecture, state_shape, n_actions):
    state_n_conv_layers, state_kernel_num, state_kernel_size, state_strides, state_conv_activation, \
    state_n_dense_layers, state_n_neurons, state_dense_activation, use_custom_net, action_n_dense_layers, \
    action_n_neurons, action_dense_activation, common_n_dense_layers, common_n_neurons, \
    common_dense_activation, use_custom_net, state_custom_net, action_custom_net, common_custom_net, \
    last_layer_activation = read_disc_net_params(net_architecture)

    stack = len(state_shape) > 1
    img_input = len(state_shape) > 2

    if use_custom_net:
        if state_custom_net is not None:
            state_model = state_custom_net(state_shape)
            state_out_size = state_model.output.shape[-1]
        else:
            if stack:
                state_model = Flatten(input_shape=state_shape)
                state_out_size = state_shape[-2] * state_shape[-1]
            else:
                state_model = _dummy_model
                state_out_size = state_shape[-1]

        if action_custom_net is not None:
            action_model = action_custom_net((n_actions,))
            action_out_size = action_model.output.shape[-1]
        else:
            action_model = _dummy_model
            action_out_size = n_actions

        common_size = state_out_size + action_out_size
        common_model = common_custom_net((common_size,))
    else:

        if not stack and not img_input:
            # Extract an integer from a tuple
            state_shape = state_shape[0]

        # build state network
        state_model = Sequential()
        if img_input:
            state_model.add(Conv2D(state_kernel_num[0], kernel_size=state_kernel_size[0], input_shape=state_shape,
                                   strides=(state_strides[0], state_strides[0]), padding='same',
                                   activation=state_conv_activation[0]))
            for i in range(1, state_n_conv_layers):
                state_model.add(Conv2D(state_kernel_num[i], kernel_size=state_kernel_size[i],
                                       strides=(state_strides[i], state_strides[i]), padding='same',
                                       activation=state_conv_activation[i]))
            state_model.add(Flatten())

        elif stack:
            state_model.add(Flatten(input_shape=state_shape))
        state_model.add(Dense(state_n_neurons[0], input_dim=state_shape, activation=state_dense_activation[0]))

        for i in range(1, state_n_dense_layers):
            state_model.add(Dense(state_n_neurons[i], activation=state_dense_activation[i]))

        # build action network
        action_model = Sequential()
        action_model.add(Dense(state_n_neurons[0], input_dim=n_actions, activation=state_dense_activation[0]))

        for i in range(1, state_n_dense_layers):
            action_model.add(Dense(state_n_neurons[i], activation=state_dense_activation[i]))

        # input_common = action_model.output.shape[-1] + state_model.output.shape[-1]
        # build common network
        common_model = Sequential()
        common_model.add(Dense(state_n_neurons[0], activation=state_dense_activation[0]))

        for i in range(1, state_n_dense_layers):
            common_model.add(Dense(state_n_neurons[i], activation=state_dense_activation[i]))

    return state_model, action_model, common_model, last_layer_activation


def _dummy_model(input):
    return input
