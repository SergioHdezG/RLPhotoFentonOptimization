import random
import time
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from src.environments.fotocaos_complete_model.modelo_completo_perox_interfaz_v2 import ModeloCinetico
import pandas as pd
# import config as conf
from ctypes import *
from src.environments.env_interfaz import EnvInterfaz, ActionSpaceInterfaz
from collections import deque
import os
from scipy.optimize import minimize
import global_enviroment_log as glob


class action_space(ActionSpaceInterfaz):
    def __init__(self, n_param):
        """
        Actions (modify k1, modify k2... modify kn)
        """
        self.low = -1.
        self.high = 1.
        self.n = n_param

class env(EnvInterfaz):
    """
    El objetivo será ajustar los parámetros k1, k2 y k3
    """

    def __init__(self, params_to_optimize=None, sodis=False, perox=False, bact=False, log_alpha_values_init=False, log_scale=False):
        np.random.seed()
        ran = np.random.rand()
        self.max_epochs = 40
        self.iterations = 0
        self.global_iter = 0
        self.n_data_division = 60

        self.error_ant = -1
        self.error_printer = 0
        self._rew = 0
        self.log_scale = log_scale

        self.log_alpha_values_init = log_alpha_values_init
        self.params_to_optimize = params_to_optimize
        self.MCinetico = ModeloCinetico(sodis=sodis, perox=perox, bact=bact, params_to_optimize=params_to_optimize,
                                        log_alpha_values_init=log_alpha_values_init)

        self.sodis = sodis
        self.perox = perox
        self.bact = bact

        # Tamaños para reutilizar modelo preentrenado en perox para bact
        if self.perox or self.bact:
            self.n_param = 5
        else:
            self.n_param = 3
        self.action_space = action_space(self.n_param)
        self.params = np.zeros(shape=(self.n_param,))
        self.params_ant = self.params

        obs_dim = self.n_param + 240 * self.perox + 240 * self.bact + 126 * self.sodis
        # Toma solo los valores coincidentes en tiempo con las medidas experiemntales en sodis y 20 valores por curva en
        # perox y bact
        # [self.curvas_perox_index, self.curvas_bact_index, self.curvas_sodis_index] = self.MCinetico.get_exp_index_v2()
        self.curv_index = False  # Falg que marca si se han inicializado self.curvas_perox_index, self.curvas_bact_index, self.curvas_sodis_index

        self.observation_space = np.zeros((obs_dim,))
        self.it_ant_nan = False

        self.stacked_count = 0  # Contador para saber si se ha estancado en una meseta el algoritmo.

        self.expert_data = self.MCinetico.get_expert_data()


        self.best_params_list = []
        self.best_params_text_list = []
        self.best_params_epi = None
        self.min_error_epi = 1e10
        self.min_global_error = 1e10
        self.err_epi_hist = []
        self.error_print = 0.

        self._dyn_reward_sodis = None
        self._dyn_reward_perox = None
        self._dyn_reward_bact = None

        self.optimization_values_queue = deque()  # Esta cola es para almacenar datos de la optimización de sodis con un
        # optimizador de estantería

        # Indices inicialización bacterias
        self.i = -10
        self.j = -10
        self.k = -10.
        self.l = -10.
        self.m = 1.

        self.sodis_grid_init = []
        for i in range(-10, 3):
            for j in range(-10, 3):
                for k in range(1, 3):
                    self.sodis_grid_init.append([np.power(10., float(i)), np.power(10., float(j)), float(k)])

    def model_mean_runtime(self):
        time_curvas = [element[0] for element in self.MCinetico.mean_runtime_list]
        time_error = [element[1] for element in self.MCinetico.mean_runtime_list]
        return np.mean(time_curvas), np.mean(time_error)

    def reset(self, test=False):
        self.iterations = 0
        self.stacked_count = 0
        self.err_epi_hist = deque(maxlen=5)

        # Inicializar parámetros hasta un punto válido
        initialized = False
        while not initialized:
            known_point_index = None
            # Parámetros por defecto
            self.params = np.zeros(shape=(self.n_param,))

            # # Parametros por defecto de bacterias
            # if self.bact:
            #     for i in range(len(self.MCinetico.params_to_optimize)):
            #         if self.MCinetico.params_to_optimize[i] in self.MCinetico.bact_pos or self.MCinetico.params_to_optimize[i] > 11:
            #             if self.MCinetico.params_to_optimize[i] < 12:
            #                 self.params[i] = 1e-6

            prob_random_init = np.random.rand()  # Probabilidad de inicialización random. si < 0.1 inicialización en
            # punto conocido
            # Buscar punto de inicio para sodis
            if self.sodis:
                pos_to_init = []
                for pos in range(len(self.MCinetico.params_to_optimize)):
                    if self.MCinetico.params_to_optimize[pos] in self.MCinetico.sodis_pos:
                        pos_to_init.append(pos)

                valid = False
                while not valid:
                    self.params, known_point_index = self._select_rand_init(pos_to_init, self.params, prob_random_init)
                    error, curvas, params = self.MCinetico.reset(self.params)
                    # self.MCinetico.render(error, curvas, self.params, 0., perox=self.perox, bact=self.bact,
                    #                       sodis=self.sodis)
                    if not self.curv_index:
                        curvas_initialization = np.copy(curvas)
                    valid = self._check_data_vality([curvas[-1]], [error[-1]], valid)
                initialized = valid
            else:
                initialized = True

            index = 0
            # Buscar punto de inicio para perox
            if self.perox and initialized:
                count = 0
                pos_to_init = []
                for pos in range(len(self.MCinetico.params_to_optimize)):
                    if self.MCinetico.params_to_optimize[pos] in self.MCinetico.perox_pos:
                        pos_to_init.append(pos)

                valid = False
                while not valid and count < 20:
                    self.params, known_point_index = self._select_rand_init(pos_to_init, self.params, prob_random_init, known_point_index)

                    # logaritmic scale params:
                    if self.log_scale:
                        for i in range(self.params.shape[0]):
                            if self.params[i] > 0:
                                # self.params[i] = np.exp(np.abs(self.params[i])-10)
                                self.params[i] = np.power(10, np.abs(self.params[i]) - 10)
                            elif self.params[i] < 0:
                                # self.params[i] = - np.exp(np.abs(self.params[i])-10)
                                self.params[i] = - np.power(10, np.abs(self.params[i]) - 10)

                    error, curvas, params = self.MCinetico.reset(self.params)
                    # self.MCinetico.render(error, curvas, self.params, 0., perox=self.perox, bact=self.bact,
                    #                       sodis=self.sodis)
                    if not self.curv_index:
                        curvas_initialization = np.copy(curvas)
                    valid = self._check_data_vality(np.array(curvas)[index], np.array(error)[index], valid)
                    if not valid:
                        # curvas[index] = np.zeros(shape=curvas[index].shape)
                        curvas = np.array(curvas)
                        # curvas[np.where(curvas <= 0.)] = 0.
                        curvas[index][np.where(np.isnan(curvas[index]))] = -1.
                        curvas[index][np.where(np.isinf(curvas[index]))] = -1.
                        curvas[index][np.where(curvas[index] > np.power(10, 10))] = -1.
                        valid = True
                    count += 1
                initialized = valid
                index += 1
            else:
                initialized = True

            # Buscar punto de inicio para bacterias
            if self.bact and initialized:
                count = 0
                pos_to_init = []
                for pos in range(len(self.MCinetico.params_to_optimize)):
                    if self.MCinetico.params_to_optimize[pos] in self.MCinetico.bact_pos:
                        pos_to_init.append(pos)

                valid = False
                while not valid and count < 50:
                    self.params, _ = self._select_rand_init(pos_to_init, self.params, prob_random_init, known_point_index)
                    # self.params, _ = self._select_rand_bact_init(pos_to_init, self.params, prob_random_init, known_point_index)
                    # aux_param = np.copy(self.params)
                    # if np.any(np.isinf(self.params)) or np.any(np.isnan(self.params)):
                    #     print()
                    # logaritmic scale params:
                    if self.log_scale:
                        for i in range(self.params.shape[0]):
                            max_i = 4 if self.bact else 5
                            if i < max_i:
                                if self.params[i] > 0:
                                    #self.params[i] = np.exp(np.abs(self.params[i])-10)
                                    self.params[i] = np.power(10, np.abs(self.params[i]) - 10)
                                elif self.params[i] < 0:
                                    #self.params[i] = - np.exp(np.abs(self.params[i])-10)
                                    self.params[i] = - np.power(10, np.abs(self.params[i]) - 10)
                    # if np.any(np.isinf(self.params)) or np.any(np.isnan(self.params)):
                    #     print()
                    error, curvas, params = self.MCinetico.reset(self.params)
                    # self.MCinetico.render(error, curvas, self.params, 0., perox=self.perox, bact=self.bact,
                    #                       sodis=self.sodis)
                    if not self.curv_index:
                        curvas_initialization = np.copy(curvas)
                    valid = self._check_data_vality(curvas[index], error[index], valid)
                    if not valid:
                        # curvas[index] = np.zeros(shape=curvas[index].shape)
                        curvas = np.array(curvas)
                        # curvas[np.where(curvas <= 0.)] = 0.
                        curvas[index][np.where(np.isnan(curvas[index]))] = -1.
                        curvas[index][np.where(np.isinf(curvas[index]))] = -1.
                        curvas[index][np.where(np.isinf(curvas[index]))] = -1.
                        curvas[index][np.where(curvas[index] > np.power(10, 10))] = -1.
                        valid = True
                initialized = valid

        curvas = np.array(curvas)
        if not self.curv_index:
            # Toma solo los valores coincidentes en tiempo con las medidas experiemntales en sodis y 20 valores por curva en
            # perox y bact
            [self.curvas_perox_index, self.curvas_bact_index, self.curvas_sodis_index] = self.MCinetico.get_exp_index_v2(curvas_initialization)
            self.curv_index = True

        self.MCinetico.close()
        self.error_printer = error
        self.error_ant = np.mean(np.array(error))

        self.params_ant = self.params
        # curvas = np.array([x[1][::self.n_data_division] for x in curvas])  # Se toma una medida de cada 300 para reducir la dimension de la observacion

        index = 0
        list_curvas = [[] for i in range(curvas.shape[0])]
        if self.perox:
            curvas_aux = []
            for i in range(self.curvas_perox_index.shape[0]):
                curvas_aux.extend(curvas[index][i][self.curvas_perox_index[i]])
            list_curvas[index] = curvas_aux
            index += 1

        if self.bact:
            curvas_aux = []
            for i in range(self.curvas_bact_index.shape[0]):
                curvas_aux.extend(curvas[index][i][self.curvas_bact_index[i]])
            list_curvas[index] = curvas_aux
            index += 1

        if self.sodis:
            curvas_aux = []
            for i in range(self.curvas_sodis_index.shape[0]):
                curvas_aux.extend(curvas[index][i][self.curvas_sodis_index[i]])
            list_curvas[index] = curvas_aux

        curvas = list_curvas
        error = -1.
        for i in range(self.err_epi_hist.maxlen):
            self.err_epi_hist.append(np.mean(np.array(error)))
        self.initial_error = np.mean(np.array(error))
        # self.Mperox._add_traj_error(error)
        # self.Mperox._add_traj(self.params)

        self._dyn_reward_sodis = self._dyn_reward_hiperbola
        self._dyn_reward_perox = self._dyn_reward_hiperbola
        self._dyn_reward_bact = self._dyn_reward_hiperbola

        self.traj_params_to_write = []

        params_to_write = "["
        for par, pos in zip(self.params, self.MCinetico.params_to_optimize):
            params_to_write += str(pos) + ": {:7.3f}, ".format(par)
        params_to_write += "]"

        string_to_write = str(self.iterations) + "\t" + str(self.error_printer) + "\t" + params_to_write\
                          + "\t0\t0\t" + str(dt.datetime.now()) + "\n"
        self.traj_params_to_write.append(string_to_write)

        curvas_state = []
        for c in curvas:
            curvas_state.extend(c)
        # print("random params init: ", self.params)
        # if np.any(np.isinf(self.params)) or np.any(np.isnan(self.params)):
        #     print()

        return np.concatenate([self.params, curvas_state])

    def _check_data_vality(self, curvas, error, valid):
        # Comprobar si todos los valores son correctos
        if not np.any(np.isnan(error)) and not np.any(np.isinf(error)):
            valid = True
            for curv in curvas:
                for c in curv:
                    if np.any(np.isnan(c)):
                        # print('isnan')
                        valid = False
                        break
                if not valid:
                    break
        return valid

    def _select_rand_init(self, pos_to_init, params, prob_random_init, random_index=None):
        if prob_random_init < 0.10 and len(self.best_params_list) > 0 and not self.sodis:
            if random_index is None:
                random_index = np.random.choice(len(self.best_params_list), 1)[0]
            # TODO: Probar que funcione
            params = self.best_params_list[random_index][0]
            # if np.any(params < -10000) or np.any(params > 10000):
            #     print()
            # if np.any(np.isinf(params)) or np.any(np.isnan(params)):
            if self.log_scale and (self.perox or self.bact):
            #     print()
                max_i = 4 if self.bact else 5
                for i in range(max_i):
                    if params[i] > 0:
                        params[i] = np.log10(np.abs(params[i]))
                    elif params[i] < 0:
                        params[i] = - np.log10(np.abs(params[i]))
            # if np.any(np.isinf(params)) or np.any(np.isnan(params)):
            #     print()

        else:
            for i in pos_to_init:
                if self.MCinetico.params_to_optimize[i] < 12:
                    if self.sodis:  #self.bact:
                        params[i] = np.random.uniform(-10, 10)
                    else:
                        params[i] = np.random.uniform(-20, 20)

                    # if self.bact:
                    #     params[i] = np.random.lognormal(1., 1., 1)
                else:  # Si el indice del parametro es 12 o 13 los valores funcionan como números enteros positivos
                    if self.bact:
                        params[i] = float(np.random.randint(1, 30))
                    elif self.sodis:
                        params[i] = float(np.random.randint(1, 15))
            # if np.any(params < -16) or np.any(params > 16):
            #     print()
            # if np.any(np.isinf(params)) or np.any(np.isnan(params)):
            #     print()
        return params, random_index

    def _select_sodis_init(self, pos_to_init, params, prob_random_init):
        index = np.random.randint(len(self.sodis_grid_init))
        params = np.array(self.sodis_grid_init[index])
        return params, None

    def _select_rand_bact_init(self, pos_to_init, params, prob_random_init, random_index=None):
        params[0] = np.power(10., self.i)
        params[1] = np.power(10., self.j)
        params[2] = np.power(10., self.k)
        params[3] = np.power(10., self.l)
        params[4] = float(self.m)
        self.m += 1.
        if self.m > 10.:
            self.m = 1.
            self.l += 1.
            if self.l > 10:
                self.l = -10
                self.k += 1.
                if self.k > 10.:
                    self.k = -10
                    self.j += 1.
                    if self.j > 10.:
                        self.j = -10
                        self.i += 1.
                        if self.i > 10.:
                            self.i = -10


        return params, random_index

    def _act(self, action):
        # for i in range(self.params.shape[0]):
        #     self.params[i] = self.params[i] + self.incremento_k * self.action_space.action_space[action][i]
        # act = np.copy(action)
        # sig = act/np.abs(act)
        # act = np.log10(np.abs(act)+1) * sig
        self.params_ant = self.params
        # self.params = self.params + np.clip(action, -1., 1.)
        if np.isnan(self.params[0]):
            print('nan params')

        if self.perox and self.bact and self.sodis:
            action = np.clip(action, -1., 1.)
            # action[4] = np.trunc(np.clip(action[4] + 15, 0., np.abs(action[4] + 15)))
            # action[11] = np.trunc(np.maximum(1., np.abs(action[11] + 15)))
            # action[12] = np.trunc(np.maximum(1., np.abs(action[11] + 15)))
            # action[4] = np.clip(action[11], 1., 30.)
            # action[4] = np.clip(action[12], 1., 30.)
        elif self.perox and self.bact:
            action = np.clip(action, -1., 1.)
            # action[4] = np.trunc(np.clip(action[4] + 15, 0., np.abs(action[4] + 15)))
            # action[9] = np.trunc(np.clip(action[9] + 15, 0., np.abs(action[9] + 15)))
            # action[9] = np.trunc(np.maximum(1., np.abs(action[9] + 15)))
            # action[4] = np.clip(action[9], 1., 30.)
        elif self.bact:
            action = np.clip(action, -1., 1.)
            # action[4] = np.trunc(np.clip(action[4]+15, 0., np.abs(action[4]+15)))
            # action[4] = np.trunc(np.maximum(1., np.abs(action[4] + 15)))
            # action[4] = np.clip(action[4], 1., 30.)

        params = self.params + np.float64(action)
        if self.bact:
            params[4] = np.clip(params[4], 1., 30.)
        # params = np.float64(action)
        if np.isnan(params[0]):
            print('nan action')

        self.params = params
        # self.params = self.params + np.clip(act, -1., 1.)
        # if aux > 0:
        #     self.params[i] = aux

        self.iterations += 1

    def step(self, action, test=False):
        # aux_action = np.copy(action)
        if self.log_scale:
            for i in range(action.shape[0]):
                max_i = 4 if self.bact else 5
                if i < max_i:
                    if action[i] > 0:
                        # action[i] = np.exp(np.abs(action[i]) - 10)
                        action[i] = np.power(10, np.abs(action[i]) - 10)
                    elif action[i] < 0:
                        # action[i] = - np.exp(np.abs(action[i]) - 10)
                        action[i] = - np.power(10, np.abs(action[i]) - 10)

        # print('action: ', str(action), ' params: ', str(self.params))
        self._act(action)
        # if np.any(self.params < -100000) or np.any(self.params > 100000):
        #     print()
        error, curvas, params = self.MCinetico.run(self.params)
        # if np.any(np.isinf(params)) or np.any(np.isnan(params)):
        #     print()
        # # controlar error minimo
        # if self.perox:
        #     error[0] = np.clip(error[0], 0.70, error[0])

        #################################################################
        #         ERROR LOG10
        ########################################################
        self.error_printer = error
        # if not np.isnan(error):
        #     error = np.log10(error + 1)
        # params = np.log10(params + 1)
        if np.any(np.isnan(self.params)) or np.any(np.isnan(params)):
            print('what????')
        # curvas = np.array([x[1][::self.n_data_division] for x in curvas])  # Se toma una medida de cada 300 para reducir la dimension de la observacion
        # curvas = np.reshape(curvas, curvas.shape[0] * curvas.shape[1])
        index = 0
        if self.perox:
            curvas_aux = []
            for i in range(self.curvas_perox_index.shape[0]):
                curvas_aux.extend(curvas[index][i][self.curvas_perox_index[i]])
            curvas[index] = curvas_aux
            index += 1

        if self.bact:
            curvas_aux = []
            for i in range(self.curvas_bact_index.shape[0]):
                curvas_aux.extend(curvas[index][i][self.curvas_bact_index[i]])
            curvas[index] = curvas_aux
            index += 1

        if self.sodis:
            curvas_aux = []
            for i in range(self.curvas_sodis_index.shape[0]):
                curvas_aux.extend(curvas[index][i][self.curvas_sodis_index[i]])
            curvas[index] = curvas_aux

        # obs = np.concatenate([[error], self.params, curvas, np.clip(action, -1., 1.)])
        # obs = np.concatenate([[error], self.params, curvas])
        curvas_state = []
        for c in curvas:
            curvas_state.extend(c)
        obs = np.concatenate([self.params, curvas_state])

        check_curvas = np.any([np.any(np.isnan(c)) or np.any(np.isinf(c)) for c in curvas])
        if np.any(np.isnan(error)) or check_curvas or np.any(np.isnan(params)) or \
           np.any(np.isinf(error)) or np.any(np.isinf(params)):
            # print('change NaN by -1')
            obs = np.where(np.isnan(obs), -1, obs)
            obs = np.where(np.isinf(obs), -1, obs)

        distancia = self.params_ant - params
        distancia = np.square(distancia)
        distancia = np.sqrt(np.sum(distancia))
        self.err_epi_hist.append(np.mean(np.array(error)))
        reward = self._reward(np.array(error), distancia)
        done = self._done(np.mean(np.array(error)))

        if np.mean(np.array(self.error_printer)) < self.min_error_epi and not np.any(np.isnan(self.error_printer)):
            self.min_error_epi = np.mean(np.array(self.error_printer))
            self.best_params_epi = params
            if np.mean(np.array(error)) < self.min_global_error:
                # TODO: creo que no hace nada
                # self.min_global_error = self.error_print
                self.min_global_error = np.mean(np.array(error))

        params_to_write = "["
        for par, pos in zip(self.params, self.MCinetico.params_to_optimize):
            params_to_write += str(pos) + ": {:7.3f}, ".format(par)
        params_to_write += "]"

        string_to_write = str(self.iterations) + "\t" + str(self.error_printer) + "\t" + params_to_write\
                          + "\t" + str(action) + "\t" + str(reward) + "\t" +\
                          str( dt.datetime.now()) + "\n"
        self.traj_params_to_write.append(string_to_write)

        best_params = None
        if done:
            result = 'Done, error: ' + str(self.error_printer) + ' [params] = ' + str(params)
            if self.best_params_epi is not None:
                if self.min_error_epi < 30000 and self.min_error_epi is not None:
                    # if np.any(self.best_params_epi < -100000) or np.any(self.best_params_epi > 100000):
                    #     print()
                    self.best_params_list.append([self.best_params_epi, self.min_error_epi, self.global_iter])

                string = ""
                for i in range(self.best_params_epi.shape[0]):
                    string += str(self.best_params_epi[i]) + "\t"
                string += str(self.min_error_epi) + "\t" + str(self.global_iter) + "\n"
                self.best_params_text_list.append(string)

                best_params = [self.best_params_epi, self.min_error_epi, self.global_iter]
                glob.best_params_text_list.append(string)
                result += ' Best error: ' + str(self.min_error_epi) + ' [Best params] = ' + str(self.best_params_epi)
                self.best_params_epi = None
                self.min_error_epi = 1e200
            if glob.test:
                glob.traj_params_to_write.append(self.traj_params_to_write)
                # self.traj_params_to_write.clear()
            if test:
                print(result)

        # if np.any(np.isinf(obs)) or np.any(np.isnan(obs)):
        #     print()

        if self.iterations >= self.max_epochs:
            self.global_iter += 1
        return obs, reward, done, best_params

    def render(self):
        error, curvas, params = self.MCinetico.run(self.params)
        self.MCinetico.render(error, curvas, params, rew=self._rew, perox=self.perox, bact=self.bact, sodis=self.sodis)

    def _reward(self, error, distancia):
        index = 0
        reward = []
        self.error_ant = []
        if self.perox:
            if np.isnan(error[index]) or np.isinf(error[index]):
                reward.append(-1.)
            else:
                reward.append(self._dyn_reward_perox(error[index], distancia))
            self.error_ant.append(error)
            index += 1
        if self.bact:
            if np.isnan(error[index]) or np.isinf(error[index]):
                reward.append(-1.)
            else:
                reward.append(self._dyn_reward_bact(error[index], distancia))
            self.error_ant.append(error)
            index += 1
        if self.sodis:
            if np.isnan(error[index]) or np.isinf(error[index]):
                reward.append(-1.)
            else:
                reward.append(self._dyn_reward_sodis(error[index], distancia))
            self.error_ant.append(error)

        reward = np.mean(reward)
        self.error_ant = np.mean(self.error_ant)
        return reward

    def _dyn_reward_hiperbola(self, error, distancia):
        return 1 /error

    def _dyn_reward_hiperbola_100(self, error, distancia):
        return 1000 /(error-100)

    # def _dyn_reward_hiperbola_100(self, error, distancia):
    #     return 10 / error

    def _done(self, error):
        # if np.isnan(error):
        #     if self.it_ant_nan:
        #         self.it_ant_nan = False
        #         return True
        #     else:
        #         self.it_ant_nan = True
        return self.iterations >= self.max_epochs #or np.isnan(error) or np.isinf(error) # or self._stop_criteria()

    def _stop_criteria(self):
        umbral = 1e-6

        if len(self.MCinetico.epi_traj_err) > 1:
            self.previous_error = self.MCinetico.epi_traj_err[-2]
            self.actual_error = self.MCinetico.epi_traj_err[-1]

            error_diff = np.abs(self.actual_error - self.previous_error)

            if error_diff < umbral and self.actual_error > 0.9:
                self.stacked_count += 1
            else:
                self.stacked_count = 0

            # print("error_diff = ", error_diff, "stacked_count = ", self.stacked_count)
            # self.error_diff_mean = self.error_diff_mean / (self.global_steps - 1) + error_diff / self.global_steps
            # print("error_diff ", error_diff)

            return self.stacked_count > 50

        else:
            return False

    def close(self):
        plt.close('all')

    # Functions for Hindsight Experience Replay
    # def calc_rew_HER(self, chain_a, chain_b):
    #     if np.array_equal(chain_a, chain_b):
    #         return 10
    #     else:
    #         return 0

    # def ismine(self):
    #     return True
    #
    # def sample_goal(self):
    #     return np.concatenate([self.experimental_model.A, self.experimental_model.B])

    def get_best_result(self):
        if len(self.best_params_list) > 0:
            best_params = np.array(self.best_params_list)
            best_params = best_params[:, 1]
            argmin = np.argmin(best_params, axis=0)
            return self.best_params_list[argmin]
        else:
            return 0

    def get_best_params(self):
        return np.array(self.best_params_list)

    def write_best_params(self, path):
        path = os.path.abspath(path)
        file = open(path, "a")
        file.writelines(self.best_params_text_list)
        file.close()

    def copy(self):
        return env(params_to_optimize=self.params_to_optimize, sodis=self.sodis, perox=self.perox, bact=self.bact, log_alpha_values_init=self.log_alpha_values_init, log_scale=self.log_scale)

    def run_sodis_optimizer(self, opt, maxiter=40, epochs=10):
        # self.reset()
        results = []
        self.error_memory = 1000000.
        # self.params_ant = np.array([-3.8, -2.32])
        # params = [0.88, -4.4, 5.05, 1.29, 9.6, -3.8, 10e-5, 10e-5, 10e-5, -2.32, 10e-5, 10e-5, 5.0]
        # params = [2.63296106, -10.01859305, -12.60452768, -10.24227304, 12.40306965, -12.19141963, -6.90527949, -6.17055679, 5.0231568, 9.41727727, 2.92414484, 5., 1.]
        # error, curvas, params = self.MCinetico.reset(params=np.array(params))

        # error2, log_error = self.calcular_error_sodis(curvas)
        # self.MCinetico.render(error, curvas, params, 0., perox=True, bact=True, sodis=True)
        # curvas_data = self.MCinetico.model.generarCurvasSodis(self.params_ant, self.action_space.n)
        # self.MCinetico.render(curvas_data[7, 0], curvas_data[:7], self.params_ant, 0., perox=False, bact=False, sodis=True)
        for i in range(epochs):
            hora = dt.datetime.now()
            print('Iter: ', i, 'hora: ', hora)
            self.reset()
            res = minimize(self.run_model_sodis, self.params, method=opt, bounds=None,
                           options={'maxiter': maxiter,
                                    # 'ftol': 1e-5,
                                    # 'gtol': 1e-05,
                                    'disp': False,
                                    # 'eps': 1.4901161193847656e-05
                                    })
            results.append([res.fun, res.x])
            print('Error: ', str(res.fun), 'Params: ', str(res.x))
        return results

    def calcular_error_sodis(self, curvas):
        error = 0.
        log_error = 0.
        for i in range(7):
            curv = curvas[2][i]
            curv = curv[self.curvas_sodis_index[i]]
            exp = self.MCinetico.sodis_expert_data[i]

            for j in range(curv.shape[0]):
                error += np.square(curv[j] - exp[1][j])
                log_error += np.square(np.log10(curv[j]) - np.log10(exp[1][j]))
        return error, log_error

    def run_model_sodis(self, params):
        # if not self.fin_optimize_flag:
        new_params = np.array(self.log_alpha_values_init)
        new_params[self.MCinetico.sodis_pos] = params
        curvas_data = self.MCinetico.model.generarCurvasSodis(new_params, self.action_space.n)
        error = self.MCinetico.calcular_error(curvas_data[:7], self.MCinetico.curvas_sodis_expert_index,
                                              self.MCinetico.sodis_expert_data)
        # error = curvas_data[7, 0]

        if error < self.error_memory:
            self.error_memory = error
        # if np.isnan(error):
        #     if not np.isnan(curvas_data.min()):
        #         print()
        return error