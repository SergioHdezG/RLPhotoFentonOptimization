import types

from RL_Agent import dqn_agent, ddqn_agent, dddqn_agent
from RL_Agent import dpg_agent, ddpg_agent
from RL_Agent import a2c_agent_discrete, a2c_agent_continuous, a2c_agent_discrete_queue, a2c_agent_continuous_queue
from RL_Agent import a3c_agent_discrete, a3c_agent_continuous
from RL_Agent import ppo_agent_discrete, ppo_agent_continuous, ppo_agent_discrete_parallel, \
    ppo_agent_continuous_parallel
from RL_Agent.base.utils import agent_globals
import pickle
import json
import numpy as np
import time
import base64
import copy
import os
import shutil
import marshal
import dill


def prueba():
    return 1


def save(agent, path):
    assert isinstance(path, str)

    folder = os.path.dirname(path)
    tmp_path = 'capoirl_tmp_saving_folder/'

    if not os.path.exists(folder) and folder != '':
        os.makedirs(folder)

    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    agent._save_network(tmp_path + 'tmp_model')
    agent_att, custom_net, actor_custom_net, critic_custom_net, common_custom_net, value_custom_net, adv_custom_net \
        = extract_agent_attributes(agent)

    custom_globals = actor_custom_globals = critic_custom_globals = common_custom_globals = value_custom_globals = adv_custom_globals = None

    if agent_att['net_architecture']['use_custom_network']:
        if custom_net is not None:
            custom_globals = dill.dumps(custom_net.__globals__)
            custom_globals = base64.b64encode(custom_globals).decode('ascii')
            custom_net = marshal.dumps(custom_net.__code__)
            custom_net = base64.b64encode(custom_net).decode('ascii')


        elif actor_custom_net is not None and critic_custom_net is not None:
            actor_custom_globals = dill.dumps(actor_custom_net.__globals__)
            actor_custom_globals = base64.b64encode(actor_custom_globals).decode('ascii')
            actor_custom_net = marshal.dumps(actor_custom_net.__code__)
            actor_custom_net = base64.b64encode(actor_custom_net).decode('ascii')

            critic_custom_globals = dill.dumps(critic_custom_net.__globals__)
            critic_custom_globals = base64.b64encode(critic_custom_globals).decode('ascii')
            critic_custom_net = marshal.dumps(critic_custom_net.__code__)
            critic_custom_net = base64.b64encode(critic_custom_net).decode('ascii')

        elif common_custom_net is not None and value_custom_net is not None and adv_custom_net is not None:
            common_custom_globals = dill.dumps(common_custom_net.__globals__)
            common_custom_globals = base64.b64encode(common_custom_globals).decode('ascii')
            common_custom_net = marshal.dumps(common_custom_net.__code__)
            common_custom_net = base64.b64encode(common_custom_net).decode('ascii')

            value_custom_globals = dill.dumps(value_custom_net.__globals__)
            value_custom_globals = base64.b64encode(value_custom_globals).decode('ascii')
            value_custom_net = marshal.dumps(value_custom_net.__code__)
            value_custom_net = base64.b64encode(value_custom_net).decode('ascii')

            adv_custom_globals = dill.dumps(adv_custom_net.__globals__)
            adv_custom_globals = base64.b64encode(adv_custom_globals).decode('ascii')
            adv_custom_net = marshal.dumps(adv_custom_net.__code__)
            adv_custom_net = base64.b64encode(adv_custom_net).decode('ascii')

    agent_att = pickle.dumps(agent_att)
    agent_att = base64.b64encode(agent_att).decode('ascii')

    try:
        f_json = tmp_path + 'tmp_model.json'
        with open(f_json, 'rb') as fp:
            json_data = fp.read()
        json_data = base64.b64encode(json_data).decode('ascii')
    except:
        json_data = None

    try:
        f_h5 = tmp_path + 'tmp_model.h5'
        with open(f_h5, 'rb') as fp:
            h5_data = fp.read()
        h5_data = base64.b64encode(h5_data).decode('ascii')
    except:
        h5_data = None

    try:
        f_h5 = tmp_path + 'tmp_modelactor.h5'
        with open(f_h5, 'rb') as fp:
            h5_actor = fp.read()
        h5_actor_data = base64.b64encode(h5_actor).decode('ascii')
        f_h5 = tmp_path + 'tmp_modelcritic.h5'
        with open(f_h5, 'rb') as fp:
            h5_critic = fp.read()
        h5_critic_data = base64.b64encode(h5_critic).decode('ascii')
    except:
        h5_actor_data = None
        h5_critic_data = None

    try:
        f_check = tmp_path + 'checkpoint'
        with open(f_check, 'rb') as fp:
            checkpoint = fp.read()
        checkpoint = base64.b64encode(checkpoint).decode('ascii')
        f_check = tmp_path + 'tmp_model.index'
        with open(f_check, 'rb') as fp:
            checkpoint_index = fp.read()
        checkpoint_index = base64.b64encode(checkpoint_index).decode('ascii')
        f_check = tmp_path + 'tmp_model.meta'
        with open(f_check, 'rb') as fp:
            checkpoint_meta = fp.read()
        checkpoint_meta = base64.b64encode(checkpoint_meta).decode('ascii')

        for file in os.listdir(tmp_path):
            if 'tmp_model.data' in file:
                f_check = tmp_path + file
        with open(f_check, 'rb') as fp:
            checkpoint_data = fp.read()
        checkpoint_data = base64.b64encode(checkpoint_data).decode('ascii')

    except:
        checkpoint = None
        checkpoint_index = None
        checkpoint_meta = None
        checkpoint_data = None

    data = {
        'agent': agent_att,
        'model_json': json_data,
        'model_h5': h5_data,
        'model_ckpt': checkpoint,
        'actor_h5': h5_actor_data,
        'critic_h5': h5_critic_data,
        'model_index': checkpoint_index,
        'model_meta': checkpoint_meta,
        'model_data': checkpoint_data,
        'custom_net': custom_net,
        'custom_globals': custom_globals,
        'actor_custom_net': actor_custom_net,
        'actor_custom_globals': actor_custom_globals,
        'critic_custom_net': critic_custom_net,
        'critic_custom_globals': critic_custom_globals,
        'common_custom_net': common_custom_net,
        'common_custom_globals': common_custom_globals,
        'value_custom_net': value_custom_net,
        'value_custom_globals': value_custom_globals,
        'adv_custom_net': adv_custom_net,
        'adv_custom_globals': adv_custom_globals
    }

    with open(path, 'w') as f:
        json.dump(data, f)

    shutil.rmtree(tmp_path)


def load(path, agent=None):
    with open(path, 'r') as f:
        data = json.load(f)

    agent_att = base64.b64decode(data['agent'])
    agent_att = pickle.loads(agent_att)

    try:
        custom_net = base64.b64decode(data['custom_net'])
        custom_globals = base64.b64decode(data['custom_globals'])
    except:
        custom_net = None
        custom_globals = None

    try:
        actor_custom_net = base64.b64decode(data['actor_custom_net'])
        actor_custom_globals = base64.b64decode(data['actor_custom_globals'])
        critic_custom_net = base64.b64decode(data['critic_custom_net'])
        critic_custom_globals = base64.b64decode(data['critic_custom_globals'])
    except:
        actor_custom_net = None
        actor_custom_globals = None
        critic_custom_net = None
        critic_custom_globals = None

    try:
        common_custom_net = base64.b64decode(data['common_custom_net'])
        common_custom_globals = base64.b64decode(data['common_custom_globals'])
        value_custom_net = base64.b64decode(data['value_custom_net'])
        value_custom_globals = base64.b64decode(data['value_custom_globals'])
        adv_custom_net = base64.b64decode(data['adv_custom_net'])
        adv_custom_globals = base64.b64decode(data['adv_custom_globals'])
    except:
        common_custom_net = None
        common_custom_globals = None
        value_custom_net = None
        value_custom_globals = None
        adv_custom_net = None
        adv_custom_globals = None

    if custom_net is not None:
        custom_globals = dill.loads(custom_globals)
        custom_globals = process_globals(custom_globals)
        code = marshal.loads(custom_net)
        custom_net = types.FunctionType(code, custom_globals, "custom_net_func")
        agent_att['net_architecture']['custom_network'] = custom_net

    elif actor_custom_net is not None and critic_custom_net is not None:
        actor_custom_globals = dill.loads(actor_custom_globals)
        actor_custom_globals = process_globals(actor_custom_globals)
        code = marshal.loads(actor_custom_net)
        actor_custom_net = types.FunctionType(code, actor_custom_globals, "actor_custom_net_func")
        agent_att['net_architecture']['actor_custom_network'] = actor_custom_net

        critic_custom_globals = dill.loads(critic_custom_globals)
        critic_custom_globals = process_globals(critic_custom_globals)
        code = marshal.loads(critic_custom_net)
        critic_custom_net = types.FunctionType(code, critic_custom_globals, "critic_custom_net_func")
        agent_att['net_architecture']['critic_custom_network'] = critic_custom_net

    elif common_custom_net is not None and value_custom_net is not None and adv_custom_net is not None:
        common_custom_globals = dill.loads(common_custom_globals)
        common_custom_globals = process_globals(common_custom_globals)
        code = marshal.loads(common_custom_net)
        common_custom_net = types.FunctionType(code, common_custom_globals, "common_custom_net_func")
        agent_att['net_architecture']['common_custom_network'] = common_custom_net

        value_custom_globals = dill.loads(value_custom_globals)
        value_custom_globals = process_globals(value_custom_globals)
        code = marshal.loads(value_custom_net)
        value_custom_net = types.FunctionType(code, value_custom_globals, "value_custom_net_func")
        agent_att['net_architecture']['value_custom_network'] = value_custom_net

        adv_custom_globals = dill.loads(adv_custom_globals)
        adv_custom_globals = process_globals(adv_custom_globals)
        code = marshal.loads(adv_custom_net)
        adv_custom_net = types.FunctionType(code, adv_custom_globals, "adv_custom_net_func")
        agent_att['net_architecture']['action_custom_network'] = adv_custom_net



    tmp_path = 'capoirl_tmp_loading_folder/'

    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    try:
        model_json_bytes = base64.b64decode(data['model_json'])
        with open(tmp_path + 'tmp_model.json', 'wb') as fp:
            fp.write(model_json_bytes)
    except:
        pass
    try:
        model_h5_bytes = base64.b64decode(data['model_h5'])
        with open(tmp_path + 'tmp_model.h5', 'wb') as fp:
            fp.write(model_h5_bytes)
    except:
        pass

    try:
        model_actor_bytes = base64.b64decode(data['actor_h5'])
        with open(tmp_path + 'tmp_modelactor.h5', 'wb') as fp:
            fp.write(model_actor_bytes)
        model_critic_bytes = base64.b64decode(data['critic_h5'])
        with open(tmp_path + 'tmp_modelcritic.h5', 'wb') as fp:
            fp.write(model_critic_bytes)
    except:
        pass

    try:
        checkpoint_bytes = base64.b64decode(data['model_ckpt'])
        with open(tmp_path + 'checkpoint', 'wb') as fp:
            fp.write(checkpoint_bytes)
        index_ckpt_bytes = base64.b64decode(data['model_index'])
        with open(tmp_path + 'tmp_model.index', 'wb') as fp:
            fp.write(index_ckpt_bytes)
        meta_ckpt_bytes = base64.b64decode(data['model_meta'])
        with open(tmp_path + 'tmp_model.meta', 'wb') as fp:
            fp.write(meta_ckpt_bytes)
        data_ckpt_bytes = base64.b64decode(data['model_data'])
        with open(tmp_path + 'tmp_model.data-00000-of-00001', 'wb') as fp:
            fp.write(data_ckpt_bytes)
    except:
        pass

    if agent is None:
        agent = create_new_agent(agent_att)

    set_agent_attributes(agent_att, agent)
    agent._load(tmp_path + 'tmp_model')

    shutil.rmtree(tmp_path)

    return agent


def extract_agent_attributes(agent):
    try:
        action_low_bound = agent.action_bound[0]
        action_high_bound = agent.action_bound[1]
    except:
        action_low_bound = None
        action_high_bound = None

    custom_net = None
    actor_custom_net = None
    critic_custom_net = None
    common_custom_net = None
    value_custom_net = None
    adv_custom_net = None
    if agent.net_architecture['use_custom_network']:
        try:
            custom_net = agent.net_architecture['custom_network']
            agent.net_architecture['custom_network'] = None
        except:
            custom_net = None
        try:
            actor_custom_net = agent.net_architecture['actor_custom_network']
            critic_custom_net = agent.net_architecture['critic_custom_network']
            agent.net_architecture['actor_custom_network'] = None
            agent.net_architecture['critic_custom_network'] = None
        except:
            actor_custom_net = None
            critic_custom_net = None
        try:
            common_custom_net = agent.net_architecture['common_custom_network']
            value_custom_net = agent.net_architecture['value_custom_network']
            adv_custom_net = agent.net_architecture['action_custom_network']
            agent.net_architecture['common_custom_network'] = None
            agent.net_architecture['value_custom_network'] = None
            agent.net_architecture['action_custom_network'] = None
        except:
            value_custom_net = None
            adv_custom_net = None
            common_custom_net = None

        if custom_net is None and actor_custom_net is None and critic_custom_net is None and common_custom_net is None and \
                value_custom_net is None and adv_custom_net is None:
            raise Exception('There are some errors when trying to save the custom network defined by the user')

    dict = {
        'state_size': agent.state_size,
        'env_state_size': agent.env_state_size,
        'n_actions': agent.n_actions,
        'stack': agent.stack,
        'learning_rate': agent.learning_rate,
        'actor_lr': agent.actor_lr,
        'critic_lr': agent.critic_lr,
        'batch_size': agent.batch_size,
        'epsilon': agent.epsilon,
        'epsilon_decay': agent.epsilon_decay,
        'epsilon_min': agent.epsilon_min,
        'gamma': agent.gamma,
        'tau': agent.tau,
        'memory_size': agent.memory_size,
        'loss_clipping': agent.loss_clipping,
        'critic_discount': agent.critic_discount,
        'entropy_beta': agent.entropy_beta,
        'lmbda': agent.lmbda,
        'train_epochs': agent.train_epochs,
        'exploration_noise': agent.exploration_noise,
        'n_step_return': agent.n_step_return,
        'n_stack': agent.n_stack,
        'img_input': agent.img_input,
        'n_parallel_envs': agent.n_parallel_envs,
        'net_architecture': agent.net_architecture,
        'action_low_bound': action_low_bound,
        'action_high_bound': action_high_bound,
        'save_base': agent.save_base,
        'save_name': agent.save_name,
        'save_each': agent.save_each,
        'save_if_better': agent.save_if_better,
        'agent_compiled': agent.agent_builded,
        'agent_name': agent.agent_name,
    }
    return dict, custom_net, actor_custom_net, critic_custom_net, common_custom_net, value_custom_net, adv_custom_net


def set_agent_attributes(att, agent):
    agent.state_size = att['state_size']
    agent.env_state_size = att['env_state_size']
    agent.n_actions = att['n_actions']
    agent.stack = att['stack']
    agent.learning_rate = att['learning_rate']
    agent.actor_lr = att['actor_lr']
    agent.critic_lr = att['critic_lr']
    agent.batch_size = att['batch_size']
    agent.epsilon = att['epsilon']
    agent.epsilon_decay = att['epsilon_decay']
    agent.epsilon_min = att['epsilon_min']
    agent.gamma = att['gamma']
    agent.tau = att['tau']
    agent.memory_size = att['memory_size']
    agent.loss_clipping = att['loss_clipping']
    agent.critic_discount = att['critic_discount']
    agent.entropy_beta = att['entropy_beta']
    agent.lmbda = att['lmbda']
    agent.train_epochs = att['train_epochs']
    agent.exploration_noise = att['exploration_noise']
    agent.n_step_return = att['n_step_return']
    agent.n_stack = att['n_stack']
    agent.img_input = att['img_input']
    agent.n_parallel_envs = att['n_parallel_envs']
    agent.net_architecture = att['net_architecture']
    agent.action_bound = [att['action_low_bound'], att['action_high_bound']]
    agent.save_base = att['save_base']
    agent.save_name = att['save_name']
    agent.save_each = att['save_each']
    agent.save_if_better = att['save_if_better']
    agent.agent_builded = att['agent_compiled']
    agent.agent_name = att['agent_name']
    return agent


def create_new_agent(att):
    if att["agent_name"] == agent_globals.names["dqn"]:
        agent = dqn_agent.Agent()
    elif att["agent_name"] == agent_globals.names["ddqn"]:
        agent = ddqn_agent.Agent()
    elif att["agent_name"] == agent_globals.names["dddqn"]:
        agent = dddqn_agent.Agent()
    elif att["agent_name"] == agent_globals.names["dpg"]:
        agent = dpg_agent.Agent()
    elif att["agent_name"] == agent_globals.names["ddpg"]:
        agent = ddpg_agent.Agent()
    elif att["agent_name"] == agent_globals.names["a2c_discrete"]:
        agent = a2c_agent_discrete.Agent()
    elif att["agent_name"] == agent_globals.names["a2c_continuous"]:
        agent = a2c_agent_continuous.Agent()
    elif att["agent_name"] == agent_globals.names["a2c_discrete_queue"]:
        agent = a2c_agent_discrete_queue.Agent()
    elif att["agent_name"] == agent_globals.names["a2c_continuous_queue"]:
        agent = a2c_agent_continuous_queue.Agent()
    elif att["agent_name"] == agent_globals.names["a3c_discrete"]:
        agent = a3c_agent_discrete.Agent()
    elif att["agent_name"] == agent_globals.names["a3c_continuous"]:
        agent = a3c_agent_continuous.Agent()
    elif att["agent_name"] == agent_globals.names["ppo_discrete"]:
        agent = ppo_agent_discrete.Agent()
    elif att["agent_name"] == agent_globals.names["ppo_discrete_parallel"]:
        agent = ppo_agent_discrete_parallel.Agent()
    elif att["agent_name"] == agent_globals.names["ppo_continuous"]:
        agent = ppo_agent_continuous.Agent()
    elif att["agent_name"] == agent_globals.names["ppo_continuous_parallel"]:
        agent = ppo_agent_continuous_parallel.Agent()
    set_agent_attributes(att, agent)

    if att['action_low_bound'] is None and att['action_high_bound'] is None:
        agent.build_agent(state_size=att["state_size"], n_actions=att["n_actions"], stack=att["stack"])
    else:
        agent.build_agent(state_size=att["state_size"], n_actions=att["n_actions"], stack=att["stack"],
                          action_bound=[att['action_low_bound'], att['action_high_bound']])
    return agent

def process_globals(custom_globals):
    globs = globals()
    for key in globs:
        for cust_key in custom_globals:
            if key == cust_key:
                custom_globals[cust_key] = globs[key]
                break
    return custom_globals

