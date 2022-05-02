def algotirhm_hyperparams(learning_rate=1e-3, batch_size=32, epsilon=1., epsilon_decay=0.9999, epsilon_min=0.1,
                          n_step_return=10):
    """
    "learning_rate": float, for example 1e-3. NN training learning rate.
    "lr_decay": float, for example 1e-3. Learning rate decay for training.
    "batch_size": int, for example 32. NN training batch size
    "epsilon": float, for example 0.1. Exploration rate.
    "epsilon_decay": float, for example 0.9999. Exploration rate decay.
    "epsilon_min": float, for example 0.15. Min exploratión rate.
    "n_step_return": int for example 10. Reward n step return.
    return: dictionary
    """
    model_params = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epsilon": epsilon,
        "epsilon_decay": epsilon_decay,
        "epsilon_min": epsilon_min,
        "n_step_return": n_step_return,
    }
    return model_params

def net_architecture(conv_layers=None, kernel_num=None, kernel_size=None, kernel_strides=None, conv_activation=None,
                     dense_layers=None, n_neurons=None, dense_activation=None, use_custom_network=False,
                     custom_network=None):
    """
    Here you can define the architecture of your model from input layer to last hidden layer. The output layer wil be
    created by the agent depending on the number of outputs and the algorithm used.
    "conv_layers": int for example 3. Number of convolutional layers.
    "kernel_num": array od ints, for example [32, 32, 64]. Number of conv kernel for each layer.
    "kernel_size": array of ints, for example [7, 5, 3]. Size of each conv kernel for each layer.
    "kernel_strides": array of ints, for example [4, 2, 2]. Stride for each conv layer.
    "conv_activation": array of string, for example ['relu', 'relu', 'relu']. Activation function for each conv layer.
    "dense_lay": int, for example 2. Number of dense layers.
    "n_neurons": array of ints, for example [1024, 1024]. Number of neurons for each dense layer.
    "dense_activation": array of string, for example ['relu', 'relu']. Activation function for each dense layer.
    "use_custom_network": Boolean. Set True if you are going to use a custom external network with your own
                          architecture. Use this together with custom_network. Default values = False
    "custom_network": Model to be used. Value based agent use keras models, for the other agents you have to return the
                      network as a tensor flow graph. These models have to be an object returned by a function.
    return: dictionary
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
                                  actor_custom_network=None, critic_custom_network=None, define_custom_output_layer=False
                                  ):
    """
    Here you can define the architecture of your model from input layer to last hidden layer. The output layer wil be
    created by the agent depending on the number of outputs and the algorithm used.
    "actor_conv_layers": int for example 3. Number of convolutional layers on agent net.
    "actor_kernel_num": array od ints, for example [32, 32, 64]. Number of conv kernel for each layer on agent net.
    "actor_kernel_size": array of ints, for example [7, 5, 3]. Size of each conv kernel for each layer on agent net.
    "actor_kernel_strides": array of ints, for example [4, 2, 2]. Stride for each conv layer on agent net.
    "actor_conv_activation": array of string, for example ['relu', 'relu', 'relu']. Activation function for each conv
                             layer on agent net.
    "actor_dense_lay": int, for example 2. Number of dense layers on agent net.
    "actor_n_neurons": array of ints, for example [1024, 1024]. Number of neurons for each dense layer on agent net.
    "actor_dense_activation": array of string, for example ['relu', 'relu']. Activation function for each dense layer
                              on agent net.
    "critic_conv_layers": int for example 3. Number of convolutional layers on critic net.
    "critic_kernel_num": array od ints, for example [32, 32, 64]. Number of conv kernel for each layer on critic net.
    "critic_kernel_size": array of ints, for example [7, 5, 3]. Size of each conv kernel for each layer on critic net.
    "critic_kernel_strides": array of ints, for example [4, 2, 2]. Stride for each conv layer on critic net.
    "critic_conv_activation": array of string, for example ['relu', 'relu', 'relu']. Activation function for each conv
                              layer on critic net.
    "critic_dense_lay": int, for example 2. Number of dense layers on critic net.
    "critic_n_neurons": array of ints, for example [1024, 1024]. Number of neurons for each dense layer on critic net.
    "critic_dense_activation": array of string, for example ['relu', 'relu']. Activation function for each dense layer
                               on critic net.
    "use_custom_network": Boolean. Set True if you are going to use a custom external network with your own
                          architecture. Use this together with actor_custom_network and critic_custom_network. Default values = False
    "actor_custom_network": Model to be used for actor. Value based agent use keras models, for the other agents you have to return the
                             network as a tensor flow graph. These models have to be an object returned by a function.
    "critic_custom_network": Model to be used for critic. Value based agent use keras models, for the other agents you have to return the
                             network as a tensor flow graph. These models have to be an object returned by a function.
    return: dictionary
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
           "define_custom_output_layer": define_custom_output_layer
           }
    return net_architecture

def save_hyperparams(base_dir, model_name, save_each, save_if_better):
    """
    "base": string, for example "/saved_models/". Folder to save the model.
    "name": string, for example "dqn_model". Name for the model.
    "save_each": int. Save the model each x iterations.
    "save_if_better": Bool. If true, save only if current mean reward value is higher than the mean reward of the last
                      model saved.
    return: dictionary
    """
    saving_model_params = {
        "base": base_dir,
        "name": model_name,
        "save_each": save_each,
        "save_if_better": save_if_better,
    }
    return saving_model_params