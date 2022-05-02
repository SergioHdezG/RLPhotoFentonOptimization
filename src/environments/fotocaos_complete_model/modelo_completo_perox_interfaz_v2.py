import random
from time import time
from ctypes import *
from numpy.ctypeslib import ndpointer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import ctypes
import os
from src.environments.env_interfaz import EnvInterfaz, ActionSpaceInterfaz

class ModeloCinetico(EnvInterfaz):
    def __init__(self, perox=True, bact=False, sodis=False, model_to_load=None, params_to_optimize=None,
                 log_alpha_values_init=False):
        # Seleccionar los parámetros a optimizar

        file = open("src/environments/fotocaos_complete_model/posiciones.txt",
                    "w")
        self.positions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        self.sodis_pos = [6, 10, 13]
        self.perox_pos = [0, 1, 2, 4, 5]
        self.bact_pos = [7, 8, 9, 11, 12]
        self.log_alpha_values_init = log_alpha_values_init
        if params_to_optimize is not None:
            params_to_optimize.sort()
            for position in params_to_optimize:
                file.write(str(position) + "\n")
        else:
            params_to_optimize = self.positions

            params_to_optimize.sort()
            for position in params_to_optimize:
                file.write(str(position) + "\n")

        file.close()

        if log_alpha_values_init:
            file = open("src/environments/fotocaos_complete_model/logAlphaValues.txt",
                        "w")
            for value in log_alpha_values_init:
                file.write(str(value) + "\n")

            file.close()

        if model_to_load is None:
            self.so_file_escribe = os.path.abspath("src/environments/fotocaos_complete_model/modelos3Completos.so")

        self.model = CDLL(self.so_file_escribe)

        # print(type(self.model))
        # print(type(self.model_escribe))

        self.n_param = len(params_to_optimize)
        self.model.loadData()

        # Modelo Cinetico Sodis
        self.model.generarCurvasSodis.argtypes = [ndpointer(dtype=c_double, shape=(self.n_param,)), c_int]
        self.model.generarCurvasSodis.restype = ndpointer(dtype=c_double, shape=(13, 7201))

        # Modelo Cinetico Perox
        self.model.generarCurvasPerox.argtypes = [ndpointer(dtype=c_double, shape=(self.n_param,)), c_int]
        self.model.generarCurvasPerox.restype = ndpointer(dtype=c_double, shape=(13, 7201))

        # Modelo Cinetico Bact
        self.model.generarCurvasBact.argtypes = [ndpointer(dtype=c_double, shape=(self.n_param,)), c_int]
        self.model.generarCurvasBact.restype = ndpointer(dtype=c_double, shape=(13, 7201))

        # self.model.return_error.restype = c_double

        self.mean_runtime_list = []
        self.curv_times = np.array([x / 60. for x in range(7201)])

        self.perox_expert_data, self.bact_expert_data, self.sodis_expert_data = self.read_exp()

        self.perox_model = perox
        self.bact_model = bact
        self.sodis_model = sodis

        if self.perox_model:
            self.params_to_optimize = self.perox_pos
        elif self.bact_model:
            self.params_to_optimize = self.bact_pos
        elif self.sodis_model:
            self.params_to_optimize = self.sodis_pos

        self.epi_traj = []
        self.epi_traj_err = []
        self.time_steps = 0

        [self.curvas_perox_expert_index, self.curvas_bact_expert_index, self.curvas_sodis_expert_index] = self.get_exp_index()
        # self.n_data = 12*7201

        self.ravel_perox_data = []
        for data in self.perox_expert_data:
            self.ravel_perox_data.extend(data[1])
        self.ravel_perox_data = np.array(self.ravel_perox_data)

        self.ravel_bact_data = []
        for data in self.bact_expert_data:
            self.ravel_bact_data.extend(data[1])
        self.ravel_bact_data = np.array(self.ravel_bact_data)

        self.ravel_sodis_data = []
        for data in self.sodis_expert_data:
            self.ravel_sodis_data.extend(data[1])
        self.ravel_sodis_data = np.array(self.ravel_sodis_data)


    def reset(self, params):
        self.epi_traj = []
        self.epi_traj_err = []
        obs = self.run(np.array(params))
        self.time_steps = 0
        return obs

    def run(self, params):
        self.param = params
        time_init = time()

        curvas = []
        error = []
        new_params = np.array(self.log_alpha_values_init)
        if self.perox_model:
            new_params[self.perox_pos] = params
            curvas_perox = np.copy(self.model.generarCurvasPerox(new_params, self.n_param))
            error_perox = curvas_perox[12][0]
            # error.append(error_perox)

        if self.bact_model:
            new_params[self.bact_pos] = params
            curvas_bact = np.copy(self.model.generarCurvasBact(new_params, self.n_param))
            error_bact = curvas_bact[12][0]
            # error.append(error_bact)

        if self.sodis_model:
            new_params[self.sodis_pos] = params
            curvas_sodis = np.copy(self.model.generarCurvasSodis(new_params, self.n_param))
            error_sodis = curvas_sodis[7][0]
            # error.append(error_sodis)

        time_model = time()

        time_extra = time()

        self.mean_runtime_list.append([time_model-time_init, time_extra-time_model])
        # print('Time model: ', time_model-time_init, 'time extra: ', time_extra-time_model, 'time total: ', time_extra-time_init)
        # curvas_data = self.model_escribe.generarCurvas(params, self.n_param)
        # curvas_data = np.ones((12, 7201))

        if self.perox_model:
            self._add_traj(params)
        #     formated_curvas_perox = []
        #     for i in range(curvas_perox.shape[0]-1):
        #         formated_curvas_perox.append(np.array([self.curv_times, curvas_perox[i]]).reshape((2, 7201)))
        #
        #     formated_curvas_perox = np.array(formated_curvas_perox)
        #     curvas.append(formated_curvas_perox)
            curvas.append(curvas_perox[:12])
            # error.append(self.calcular_error(curvas_perox[:12], self.curvas_perox_expert_index, self.perox_expert_data))
            # error_perox = self.calcular_error(curvas_perox[:12], self.curvas_perox_expert_index, self.perox_expert_data)
            error.append(error_perox)

        if self.bact_model:
            self._add_traj(params)
        #     formated_curvas_bact = []
        #     for i in range(curvas_bact.shape[0] - 1):
        #         formated_curvas_bact.append(np.array([self.curv_times, curvas_bact[i]]).reshape((2, 7201)))
        #
        #     formated_curvas_bact = np.array(formated_curvas_bact)
        #      curvas.append(formated_curvas_bact)
            curvas.append(curvas_bact[:12])
            # error.append(self.calcular_error(curvas_bact[:12], self.curvas_bact_expert_index, self.bact_expert_data))
            error.append(error_bact)

        if self.sodis_model:
            self._add_traj(params)
            # formated_curvas_sodis = []
            # for i in range(7):
            #     formated_curvas_sodis.append(np.array([self.curv_times, curvas_sodis[i]]).reshape((2, 7201)))
            #
            # formated_curvas_sodis = np.array(formated_curvas_sodis)
            # curvas.append(formated_curvas_sodis)
            curvas.append(curvas_sodis[:7])
            error.append(self.calcular_error(curvas_sodis[:7], self.curvas_sodis_expert_index, self.sodis_expert_data))

        self._add_traj_error(np.mean(np.array(error)))
        self.time_steps += 1

        return error, curvas, self.param

    def render(self, error, curvas, param, rew, perox=False, bact=False, sodis=False):
        """
        :error: array of 1 to 3 positions. If perox = bact = sodis = True then [perox_data, bact_data, sodis_data],
                if perox = False and bact = sodis = True then [bact_data, sodis_data]
                if bact = False and perox = sodis = True then [perox_data, sodis_data]
                if sodis = False and perox = bact = True then [perox_data, bact_data]
                if perox = True and bact = sodis = False then [perox_data]
                if bact = True and perox = sodis = False then [bact_data]
                if sodis = True and perox = bact = False then [sodis_data]
        :curvas: array of 1 to 3 positions. If perox = bact = sodis = True then [perox_data, bact_data, sodis_data],
                if perox = False and bact = sodis = True then [bact_data, sodis_data]
                if bact = False and perox = sodis = True then [perox_data, sodis_data]
                if sodis = False and perox = bact = True then [perox_data, bact_data]
                if perox = True and bact = sodis = False then [perox_data]
                if bact = True and perox = sodis = False then [bact_data]
                if sodis = True and perox = bact = False then [sodis_data]
        :params: array of 1 to 3 positions. If perox = bact = sodis = True then [perox_data, bact_data, sodis_data],
                if perox = False and bact = sodis = True then [bact_data, sodis_data]
                if bact = False and perox = sodis = True then [perox_data, sodis_data]
                if sodis = False and perox = bact = True then [perox_data, bact_data]
                if perox = True and bact = sodis = False then [perox_data]
                if bact = True and perox = sodis = False then [bact_data]
                if sodis = True and perox = bact = False then [sodis_data]
        :rew: float
        :perox: bool
        :bact: bool
        :sodis: bool
        """
        if not perox and not bact and not sodis:
            raise Exception('perox, bact and sodis are set to False. At least one of them must be set to true.')
        array_index = 0
        plt.clf()

        fig = plt.figure(1)

        if perox:
            curv_times = []
            for i in range(12):
                last_index = np.where(curvas[array_index][i] == -1.)[0]
                if len(last_index) > 0:
                    curv_times.append(self.curv_times[:last_index[0]])
                else:
                    curv_times.append(self.curv_times)
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.set(xlabel='Tiempo', ylabel='Concentracion',
                    title='Perox Model')
            ax1.grid()
            self.plot_data(ax1, curvas[array_index], self.perox_expert_data, curv_times, yscale='linear')
            array_index += 1
            # plt.legend(
            #     ('Exp 1', 'Exp 2', 'Exp 3', 'Exp 4', 'Exp 5', 'Exp 6', 'Exp 7', 'Exp 8', 'Exp 9', 'Exp 10', 'Exp 11',
            #      'Exp 12'),
            #     loc='upper right')
        if bact:
            curv_times = []
            for i in range(12):
                last_index = np.where(curvas[array_index][i] == -1.)[0]
                if len(last_index) > 0:
                    curv_times.append(self.curv_times[:last_index[0]])
                else:
                    curv_times.append(self.curv_times)
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.set(xlabel='Tiempo', ylabel='Concentracion',
                    title='Bact Model')
            ax2.grid()
            self.plot_data(ax2, curvas[array_index], self.bact_expert_data, curv_times, yscale='log')
            array_index += 1
            # plt.legend(
            #     ('Exp 1', 'Exp 2', 'Exp 3', 'Exp 4', 'Exp 5', 'Exp 6', 'Exp 7', 'Exp 8', 'Exp 9', 'Exp 10', 'Exp 11',
            #      'Exp 12'),
            #     loc='upper right')
        if sodis:
            curv_times = [self.curv_times for i in range(7)]
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.set(xlabel='Tiempo', ylabel='Concentracion',
                    title='Sodis Model')
            ax3.grid()
            self.plot_data(ax3, curvas[array_index], self.sodis_expert_data, curv_times, yscale='log')
            array_index += 1
            # plt.legend(
            #     ('Exp 1', 'Exp 2', 'Exp 3', 'Exp 4', 'Exp 5', 'Exp 6', 'Exp 7'),
            #     loc='upper right')

        if rew > 0:
            font = {'color': 'green'}
        else:
            font = {'color': 'red'}

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set(xlabel='Steps', ylabel='Value',
                title='Params')

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', "lime", "orange", "indianred", "indigo", "skyblue", "olive", "gray"]
        for i in range(len(param)):
            x = range(int(self.time_steps/2)+1)  # Entre dos por que al renderizar se dobla la cuenta

            ax4.text((self.time_steps/2)-1.25, param[i]+0.2, 'k(' + str(i) + ")= {:.1f}".format(param[i]))
            y = np.array(self.epi_traj)[::2, i]
            ax4.plot(x, y, marker=".", linestyle="--", color=colors[i])

        plt.draw()

        plt.pause(10e-50)

    def plot_data(self, subplot, curvas, exp_data, curv_times, yscale):

        subplot.plot(exp_data[0][0], exp_data[0][1], ".b")
        subplot.plot(exp_data[1][0], exp_data[1][1], ".g")
        subplot.plot(exp_data[2][0], exp_data[2][1], ".r")
        subplot.plot(exp_data[3][0], exp_data[3][1], ".c")
        subplot.plot(exp_data[4][0], exp_data[4][1], ".m")
        subplot.plot(exp_data[5][0], exp_data[5][1], ".y")
        subplot.plot(exp_data[6][0], exp_data[6][1], ".k")
        if curvas.shape[0] > 7:
            subplot.plot(exp_data[7][0], exp_data[7][1], marker=".", color="lime", linestyle="None")
            subplot.plot(exp_data[8][0], exp_data[8][1], marker=".", color="orange", linestyle="None")
            subplot.plot(exp_data[9][0], exp_data[9][1], marker=".", color="indianred", linestyle="None")
            subplot.plot(exp_data[10][0], exp_data[10][1], marker=".", color="indigo", linestyle="None")
            subplot.plot(exp_data[11][0], exp_data[11][1], marker=".", color="skyblue", linestyle="None")

        alpha = 1.0
        subplot.plot(curv_times[0], curvas[0][:curv_times[0].shape[0]], "-b", alpha=alpha)
        subplot.plot(curv_times[1], curvas[1][:curv_times[1].shape[0]], "-g", alpha=alpha)
        subplot.plot(curv_times[2], curvas[2][:curv_times[2].shape[0]], "-r", alpha=alpha)
        subplot.plot(curv_times[3], curvas[3][:curv_times[3].shape[0]], "-c", alpha=alpha)
        subplot.plot(curv_times[4], curvas[4][:curv_times[4].shape[0]], "-m", alpha=alpha)
        subplot.plot(curv_times[5], curvas[5][:curv_times[5].shape[0]], "-y", alpha=alpha)
        subplot.plot(curv_times[6], curvas[6][:curv_times[6].shape[0]], "-k", alpha=alpha)
        if curvas.shape[0] > 7:
            subplot.plot(curv_times[7], curvas[7][:curv_times[7].shape[0]], color="lime", linestyle="solid", alpha=alpha)
            subplot.plot(curv_times[8], curvas[8][:curv_times[8].shape[0]], color="orange", linestyle="solid", alpha=alpha)
            subplot.plot(curv_times[9], curvas[9][:curv_times[9].shape[0]], color="indianred", linestyle="solid", alpha=alpha)
            subplot.plot(curv_times[10], curvas[10][:curv_times[10].shape[0]], color="indigo", linestyle="solid", alpha=alpha)
            subplot.plot(curv_times[11], curvas[11][:curv_times[11].shape[0]], color="skyblue", linestyle="solid", alpha=alpha)

        subplot.set_yscale(yscale)

    def close(self):
        plt.close(1)

    def _check_curv_expdata(self, curv_data, exp_data):
        index_aux = []
        rang = curv_data.shape[0]
        for experimento in range(len(exp_data)):
            i = 0
            indices = []
            for n in exp_data[experimento][0]:
                n_min = n - 0.01
                n_max = n + 0.01
                for i in range(i, rang):
                    e = curv_data[i]
                    if n_min < e < n_max:
                        indices.append(i)
                        break

            index_aux.append(indices)

        return index_aux

    def get_exp_index(self):
        """
        Este método devuelve los valores de las curvas correspondientes a los índices donde hay valores experiementales.
        """
        index = 0
        nuevos_index = []

        index_aux = self._check_curv_expdata(self.curv_times, self.perox_expert_data)
        index += 1
        nuevos_index.append(np.array(index_aux))

        index_aux = self._check_curv_expdata(self.curv_times, self.bact_expert_data)
        index += 1
        nuevos_index.append(np.array(index_aux))

        index_aux = self._check_curv_expdata(self.curv_times, self.sodis_expert_data)
        nuevos_index.append(np.array(index_aux))

        return nuevos_index

    def _check_curv_expdata_v2(self, curv_data, exp_data):

        index_aux = []
        for experimento in range(len(exp_data)):
            max_index = 7200
            # last_time_data = exp_data[experimento][0][-1]
            no_data = np.where(curv_data[experimento] == -1.)[0]
            if len(no_data) > 0:
                max_index = no_data[0]-1
            indices = np.linspace(0, max_index, num=20, dtype=np.int)
            index_aux.append(indices)

        return index_aux


    def get_exp_index_v2(self, curvas):
        """
        Este método devuelve 20 valores de cada curva.
        """
        index = 0
        nuevos_index = []

        if self.perox_model:
            index_aux = self._check_curv_expdata_v2(curvas[index], self.perox_expert_data)
            index += 1
            nuevos_index.append(np.array(index_aux))
        else:
            nuevos_index.append(np.array([]))

        if self.bact_model:
            index_aux = self._check_curv_expdata_v2(curvas[index], self.bact_expert_data)
            index += 1
            nuevos_index.append(np.array(index_aux))
        else:
            nuevos_index.append(np.array([]))

        if self.sodis_model:
            index_aux = self._check_curv_expdata(self.curv_times, self.sodis_expert_data)
            nuevos_index.append(np.array(index_aux))
        else:
            nuevos_index.append(np.array([]))

        [self.curvas_perox_index, self.curvas_bact_index, self.curvas_sodis_index] = nuevos_index
        return nuevos_index

    def read_exp(self):
        # leer datos de fichero
        perox_data = []
        bact_data = []
        sodis_data = []

        for i in range(12):
            f = open("src/environments/fotocaos_complete_model/DatosPerox/DatosPerox"+str(i+1)+".txt", "r")
            list = []

            for x in f:
                datos = x.split("\t")
                if len(datos) == 2:
                    list.append(datos)

            list = np.array(list).transpose()

            tp_tuple = np.zeros(shape=(2, len(list[0])))
            for j in range(list.shape[1]):
                tp_tuple[0][j] = float(list[0][j])
                tp_tuple[1][j] = float(list[1][j])

            perox_data.append(tp_tuple)

        for i in range(12):
            f = open("src/environments/fotocaos_complete_model/DatosBact/DatosExp"+str(i+1)+".txt", "r")
            list = []

            for x in f:
                datos = x.split("\t")
                if len(datos) == 2:
                    list.append(datos)

            list = np.array(list).transpose()

            tp_tuple = np.zeros(shape=(2, len(list[0])))
            for j in range(list.shape[1]):
                tp_tuple[0][j] = float(list[0][j])
                tp_tuple[1][j] = float(list[1][j])

            bact_data.append(tp_tuple)

        for i in range(7):
            f = open("src/environments/fotocaos_complete_model/SodisData/DatosExp"+str(i+1)+".txt", "r")
            list = []

            for x in f:
                datos = x.split("\t")
                if len(datos) == 2:
                    list.append(datos)

            list = np.array(list).transpose()

            tp_tuple = np.zeros(shape=(2, len(list[0])))
            for j in range(list.shape[1]):
                tp_tuple[0][j] = float(list[0][j])
                tp_tuple[1][j] = float(list[1][j])

            sodis_data.append(tp_tuple)

        return perox_data, bact_data, sodis_data


    def read_curv(self):
        list_curvas = []

        for i in range(12):
            f = open("src/environments/fotocaos_complete_model/DatosPerox" + str(i + 1) + ".txt", "r")
            list = []
            for x in f:
                x = x.replace(",", ".")
                datos = x.split("\t")
                if len(datos) == 2:
                    list.append(datos)

            list = np.array(list).transpose()

            tp_tuple = np.zeros(shape=(2, len(list[0])))
            for j in range(list.shape[1]):
                tp_tuple[0][j] = float(list[0][j])
                tp_tuple[1][j] = float(list[1][j])

            list_curvas.append(tp_tuple)
        return np.array(list_curvas)

    def _add_traj(self, params):
        self.epi_traj.append(np.copy(params))
        return self.epi_traj

    def _add_traj_error(self, error):
        return self.epi_traj_err.append(error)

    def get_expert_data(self):
        perox_expert_data = [dat[1] for dat in self.perox_expert_data]
        p_data = []
        for d in perox_expert_data:
            p_data = np.concatenate((p_data, d))

        bact_expert_data = [dat[1] for dat in self.bact_expert_data]
        b_data = []
        for d in bact_expert_data:
            b_data = np.concatenate((b_data, d))

        sodis_expert_data = [dat[1] for dat in self.sodis_expert_data]
        s_data = []
        for d in sodis_expert_data:
            s_data = np.concatenate((s_data, d))

        return [p_data, b_data, s_data]

    def calcular_error(self, curvas, indexes, expert_data):
        log_error = 0.
        for i in range(len(indexes)):
            curv = curvas[i]
            curv = curv[indexes[i]]
            exp = expert_data[i]+1
            if np.any(np.isnan(curv)) or np.any(np.isinf(curv)):
                log_error += np.nan
            else:
                curv[curv <= 0.] = 1e-5
                for j in range(curv.shape[0]):
                    log_error += np.square(np.log10(curv[j]) - np.log10(exp[1][j]))

        # Cuando el error es excesivamente grande lo consideramos infinito por que se trata de un punto de
        # inicialización no válido
        # if log_error > 10000:
        #     log_error = np.nan
        return log_error