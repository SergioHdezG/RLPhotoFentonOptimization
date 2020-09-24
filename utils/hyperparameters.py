import multiprocessing

def algotirhm_hyperparams(learning_rate=1e-3, batch_size=32, epsilon=1., epsilon_decay=0.9999, epsilon_min=0.1,
                          n_step_return=10, n_steps_update=10, buffer_size=2048, gamma=0.95, tau=0.001,
                          loss_clipping=0.2, critic_discount=0.5, entropy_beta=0.001, ppo_lmbda=0.95,
                          ppo_train_epochs=10, exploration_noise=1.0, n_parallel_envs=multiprocessing.cpu_count()):
    """
    :param learning_rate:   float, for example 1e-3. NN training learning rate.
    :param lr_decay:        float, for example 1e-3. Learning rate decay for training.
    :param batch_size:      int, for example 32. NN training batch size
    :param epsilon:         float, for example 0.1. Exploration rate.
    :param epsilon_decay:   float, for example 0.9999. Exploration rate decay.
    :param epsilon_min:     float, for example 0.15. Min exploratión rate.
    :param n_step_return:   int for example 10. Reward n step return.
    :return: dictionary
    """
    model_params = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epsilon": epsilon,
        "epsilon_decay": epsilon_decay,
        "epsilon_min": epsilon_min,
        "n_step_return": n_step_return,
        "n_steps_update": n_steps_update,
        "buffer_size": buffer_size,
        "gamma": gamma,
        "loss_clipping": loss_clipping,
        "critic_discount": critic_discount,
        "entropy_beta": entropy_beta,
        "lmbda": ppo_lmbda,
        "ppo_train_epochs": ppo_train_epochs,
        "exploration_noise": exploration_noise,
        "n_parallel_envs": n_parallel_envs,
        "tau": tau,
    }
    return model_params

def net_architecture(conv_layers=None, kernel_num=None, kernel_size=None, kernel_strides=None, conv_activation=None,
                     dense_layers=None, n_neurons=None, dense_activation=None, use_custom_network=False,
                     custom_network=None):
    """
    Here you can define the architecture of your model from input layer to last hidden layer. The output layer wil be
    created by the agent depending on the number of outputs and the algorithm used.
    :param conv_layers:         int for example 3. Number of convolutional layers.
    :param kernel_num:          array od ints, for example [32, 32, 64]. Number of conv kernel for each layer.
    :param kernel_size:         array of ints, for example [7, 5, 3]. Size of each conv kernel for each layer.
    :param kernel_strides:      array of ints, for example [4, 2, 2]. Stride for each conv layer.
    :param conv_activation:     array of string, for example ['relu', 'relu', 'relu']. Activation function for each conv
                                layer.
    :param dense_lay:           int, for example 2. Number of dense layers.
    :param n_neurons:           array of ints, for example [1024, 1024]. Number of neurons for each dense layer.
    :param dense_activation:    array of string, for example ['relu', 'relu']. Activation function for each dense layer.
    :param use_custom_network:  boolean. Set True if you are going to use a custom external network with your own
                                architecture. Use this together with custom_network. Default values = False
    :param custom_network:      Model to be used. Value based agent use keras models, for the other agents you have to
                                return the network as a tensor flow graph. These models have to be an object returned by
                                a function.
    :return: dictionary
    """
    net_architecture = {
        "conv_layers": conv_layers,
        "kernel_num": kernel_num,
        "kernel_size": kernel_size,
        "kernel_strides": kernel_strides,
        "conv_activation": conv_activation,
        "dense_lay": dense_layers,
        "n_neurons": n_neurons,
        "dense_activation": dense_activation,
        "use_custom_network": use_custom_network,
        "custom_network": custom_network,
    }
    return net_architecture

def actor_critic_net_architecture(actor_conv_layers=None, actor_kernel_num=None, actor_kernel_size=None,
                                  actor_kernel_strides=None, actor_conv_activation=None, actor_dense_layers=None,
                                  actor_n_neurons=None, actor_dense_activation=None,
                                  critic_conv_layers=None, critic_kernel_num=None, critic_kernel_size=None,
                                  critic_kernel_strides=None, critic_conv_activation=None, critic_dense_layers=None,
                                  critic_n_neurons=None, critic_dense_activation=None, use_custom_network=False,
                                  actor_custom_network=None, critic_custom_network=None
                                  ):
    """
    Here you can define the architecture of your model from input layer to last hidden layer. The output layer wil be
    created by the agent depending on the number of outputs and the algorithm used.
    :param actor_conv_layers:       int for example 3. Number of convolutional layers on agent net.
    :param actor_kernel_num:        array od ints, for example [32, 32, 64]. Number of conv kernel for each layer on
                                    agent net.
    :param actor_kernel_size:       array of ints, for example [7, 5, 3]. Size of each conv kernel for each layer on
                                    agent net.
    :param actor_kernel_strides:    array of ints, for example [4, 2, 2]. Stride for each conv layer on agent net.
    :param actor_conv_activation:   array of string, for example ['relu', 'relu', 'relu']. Activation function for each
                                    conv layer on agent net.
    :param actor_dense_layers:      int, for example 2. Number of dense layers on agent net.
    :param actor_n_neurons:         array of ints, for example [1024, 1024]. Number of neurons for each dense layer on
                                    agent net.
    :param actor_dense_activation:  array of string, for example ['relu', 'relu']. Activation function for each dense
                                    layer on agent net.
    :param critic_conv_layers:      int for example 3. Number of convolutional layers on critic net.
    :param critic_kernel_num:       array od ints, for example [32, 32, 64]. Number of conv kernel for each layer on
                                    critic net.
    :param critic_kernel_size:      array of ints, for example [7, 5, 3]. Size of each conv kernel for each layer on
                                    critic net.
    :param critic_kernel_strides:   array of ints, for example [4, 2, 2]. Stride for each conv layer on critic net.
    :param critic_conv_activation:  array of string, for example ['relu', 'relu', 'relu']. Activation function for each
                                    conv layer on critic net.
    :param critic_dense_layers:     int, for example 2. Number of dense layers on critic net.
    :param critic_n_neurons:        array of ints, for example [1024, 1024]. Number of neurons for each dense layer on
                                    critic net.
    :param critic_dense_activation: array of string, for example ['relu', 'relu']. Activation function for each dense
                                    layer on critic net.
    :param use_custom_network:      boolean. Set True if you are going to use a custom external network with your own
                                    architecture. Use this together with actor_custom_network and critic_custom_network.
                                    Default values = False
    :param actor_custom_network:    Model to be used for actor. Value based agent use keras models, for the other agents
                                    you have to return the network as a tensor flow graph. These models have to be an
                                    object returned by a function.
    :param critic_custom_network:   Model to be used for critic. Value based agent use keras models, for the other
                                    agents you have to return the network as a tensor flow graph. These models have to
                                    be an object returned by a function.
    :return: dictionary
    """
    net_architecture = {
           "actor_conv_layers": actor_conv_layers,
           "actor_kernel_num": actor_kernel_num,
           "actor_kernel_size": actor_kernel_size,
           "actor_kernel_strides": actor_kernel_strides,
           "actor_conv_activation": actor_conv_activation,
           "actor_dense_lay": actor_dense_layers,
           "actor_n_neurons": actor_n_neurons,
           "actor_dense_activation": actor_dense_activation,

           "critic_conv_layers": critic_conv_layers,
           "critic_kernel_num": critic_kernel_num,
           "critic_kernel_size": critic_kernel_size,
           "critic_kernel_strides": critic_kernel_strides,
           "critic_conv_activation": critic_conv_activation,
           "critic_dense_lay": critic_dense_layers,
           "critic_n_neurons": critic_n_neurons,
           "critic_dense_activation": critic_dense_activation,

           "use_custom_network": use_custom_network,
           "actor_custom_network": actor_custom_network,
           "critic_custom_network": critic_custom_network,
           }
    return net_architecture

def save_hyperparams(base_dir, model_name, save_each, save_if_better):
    """
    :param base_dir:        string, for example "/saved_models/". Folder to save the model.
    :param model_name:      string, for example "dqn_model". Name for the model.
    :param save_each:       int. Save the model each x iterations.
    :param save_if_better:  bool. If true, save only if current mean reward value is higher than the mean reward of the
                            last model saved.
    :return dictionary
    """
    saving_model_params = {
        "base": base_dir,
        "name": model_name,
        "save_each": save_each,
        "save_if_better": save_if_better,
    }
    return saving_model_params

def irl_hyperparams(lr_disc=1e-6, batch_size_disc=128, epochs_disc=5, val_split_disc=0.2, agent_collect_iter=10,
                    agent_train_iter=100):
    """
    :param lr_disc:             float. Discriminator NN training learning rate.
    :param batch_size_disc:     int. Discriminator NN training batch size.
    :param epochs_disc:         int. Epoch for training discriminator NN on each iteration.
    :param val_split_disc:      float. Validation split for the data when training the discriminator NN.
    :param agent_collect_iter:  int. Number of iterations when agent is collecting data in each iteration of Vanilla
                                Deep IRL.
    :param agent_train_iter:    int. Number of iterations when agent is training in each iteration of Vanilla  Deep IRL.
    :return                    dictionary.
    """
    model_params = {
        "lr_disc": lr_disc,
        "batch_size_disc": batch_size_disc,
        "epochs_disc": epochs_disc,
        "val_split_disc": val_split_disc,
        "agent_collect_iter": agent_collect_iter,
        "agent_train_iter": agent_train_iter,
    }
    return model_params

def irl_discriminator_net_architecture(state_conv_layers=None, state_kernel_num=None, state_kernel_size=None,
                                       state_kernel_strides=None, state_conv_activation=None, state_dense_lay=None,
                                       state_n_neurons=None, state_dense_activation=None,
                                       action_dense_lay=None, action_n_neurons=None, action_dense_activation=None,
                                       common_dense_lay=None, common_n_neurons=None, common_dense_activation=None,
                                       use_custom_network=None, state_custom_network=None, action_custom_network=False,
                                       common_custom_network=None, last_layer_activation=None
                                       ):
    """
    Here you can define the architecture of your model from input layer to last hidden layer. The output layer wil be
    created by the agent depending on the number of outputs and the algorithm used.
    :param state_conv_layers:       int for example 3. Number of convolutional layers on agent net.
    :param state_kernel_num:        array od ints, for example [32, 32, 64]. Number of conv kernel for each layer on
                                    agent net.
    :param state_kernel_size:       array of ints, for example [7, 5, 3]. Size of each conv kernel for each layer on
                                    agent net.
    :param state_kernel_strides:    array of ints, for example [4, 2, 2]. Stride for each conv layer on agent net.
    :param state_conv_activation:   array of string, for example ['relu', 'relu', 'relu']. Activation function for each
                                    conv layer on agent net.
    :param state_dense_lay:      int, for example 2. Number of dense layers on agent net.
    :param state_n_neurons:         array of ints, for example [1024, 1024]. Number of neurons for each dense layer on
                                    agent net.
    :param state_dense_activation:  array of string, for example ['relu', 'relu']. Activation function for each dense
                                    layer on agent net.
    :param action_dense_lay:      int for example 3. Number of convolutional layers on critic net.
    :param action_n_neurons:       array od ints, for example [32, 32, 64]. Number of conv kernel for each layer on
                                    critic net.
    :param action_dense_activation:      array of ints, for example [7, 5, 3]. Size of each conv kernel for each layer on
                                    critic net.
    :param common_dense_lay:   array of ints, for example [4, 2, 2]. Stride for each conv layer on critic net.
    :param common_n_neurons:  array of string, for example ['relu', 'relu', 'relu']. Activation function for each
                                    conv layer on critic net.
    :param common_dense_activation:     int, for example 2. Number of dense layers on critic net.
    :param use_custom_network:          boolean. Set True if you are going to use a custom external network with your own
                                    architecture. Use this together with actor_custom_network and critic_custom_network.
                                    Default values = False
    :param state_custom_network: array of string, for example ['relu', 'relu']. Activation function for each dense
                                    layer on critic net.
    :param action_custom_network:
    :param common_custom_network:    Model to be used for actor. Value based agent use keras models, for the other agents
                                    you have to return the network as a tensor flow graph. These models have to be an
                                    object returned by a function.
    :param last_layer_activation:   Model to be used for critic. Value based agent use keras models, for the other
                                    agents you have to return the network as a tensor flow graph. These models have to
                                    be an object returned by a function.
    :return: dictionary
    """
    net_architecture = {
           "state_conv_layers": state_conv_layers,
           "state_kernel_num": state_kernel_num,
           "state_kernel_size": state_kernel_size,
           "state_kernel_strides": state_kernel_strides,
           "state_conv_activation": state_conv_activation,

           "state_dense_lay": state_dense_lay,
           "state_n_neurons": state_n_neurons,
           "state_dense_activation": state_dense_activation,

           "action_dense_lay": action_dense_lay,
           "action_n_neurons": action_n_neurons,
           "action_dense_activation": action_dense_activation,

           "common_dense_lay": common_dense_lay,
           "common_n_neurons": common_n_neurons,
           "common_dense_activation": common_dense_activation,

           "use_custom_network": use_custom_network,
           "state_custom_network": state_custom_network,
           "action_custom_network": action_custom_network,
           "common_custom_network": common_custom_network,

           "last_layer_activation": last_layer_activation,
           }
    return net_architecture