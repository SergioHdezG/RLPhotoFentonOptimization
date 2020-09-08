import pandas as pd
import numpy as np

data  = pd.read_csv("/home/serch/TFM/IRL3/experimentos_result/carla_stop/gail_stop.txt", sep="\t")
# data = data[83:-1]
data = data[54:-1]

reward = []
camino_recorrido = []
desplaz_abs = []
speed = []
fps = []
epochs = []

salidas_carril = []
carril_aux = []
metricas = []
rl = []

for d in data.values:
    list = d[0].split()
    if "fuera" in d[0]:
        carril_aux.append(float(list[3]))
    if "Episode" in d[0]:
        epochs.append(float(list[3]))
        reward.append(float(list[5]))
        rl.append(list)
    if "RL" in d[0]:
        camino_recorrido.append(float(list[5]))
        desplaz_abs.append(float(list[8]))
        speed.append(float(list[11]))
        fps.append(float(list[14]))
        metricas.append(list)
        salidas_carril.append(carril_aux)
        carril_aux = []

mean_reward = np.mean(reward)
mean_epochs = np.mean(epochs)
mean_camino_rec = np.mean(camino_recorrido)
mean_desplaz_abs = np.mean(desplaz_abs)
mean_speed = np.mean(speed)
mean_fps = np.mean(fps)

print(data)
