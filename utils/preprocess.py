import numpy as np

def preprocess_perox(obs):
    obs[0] = obs[0]
    obs[1:-3] = obs[1:-3]/35
    obs[-3:] = obs[-3:]/20

    return obs


def preprocess_perox_act(obs):
    obs[0] = obs[0]
    obs[1:-3] = obs[1:-3]/35
    obs[-3:] = np.clip(obs[-3:], -1., 1.)

    return obs

def preprocess_perox_glob_pos(obs):
    obs[0] = obs[0]
    obs[1:4] = obs[1:4] / 35
    obs[4:] = obs[4:]/35
    return obs

def perox_only_norm_hind_linear_legacy(obs):
    obs[:3] = obs[:3] / 20.  # Normalizamos números entre [-20, 20], si sale de ahí se asume que puede valer mas que 1 o
    # menos que -1.
    obs[3:][obs[3:] <= 0.] = 1e-5
    obs[3:] = np.log10(obs[3:])  # Lo pasamos a escala logarítmica

    indices = [(3, 17), (17, 31), (31, 42), (42, 53), (53, 64), (64, 75), (75, 89), (89, 103), (103, 117),
               (117, 131), (131, 145), (145, 159), (159, 173), (173, 187), (187, 198), (198, 209), (209, 220),
               (220, 231), (231, 245), (245, 259), (259, 273), (273, 287), (287, 301), (301, 315)]
    for i in range(24):
        obs[indices[i][0]:indices[i][1]] = \
            obs[indices[i][0]:indices[i][1]] / np.abs(obs[indices[i][0]])  # Normalizamos cada curva entre 0 y 1.
        obs[indices[i][0]:indices[i][1]] = np.clip(obs[indices[i][0]:indices[i][1]], -1., np.max(obs[indices[i][0]:indices[i][1]]))
    return obs

def preprocess_perox_glob_pos_noerr(obs):
    obs[:4] = obs[:4] / 35
    obs[4:] = obs[4:]/35
    return obs

def preprocess_perox_desp_local(obs):
    obs[0] = 1/obs[0]
    obs[1:-3] = obs[1:-3]/35
    obs[-3:] = obs[-3:]/35

    return obs

def preprocess_perox_glob_legacy(obs):
    obs[:4] = obs[:4] / 20.
    obs[4:] = obs[4:]/32.27
    return obs
####################### Preprocesado para modelo completo ###############################3

def perox_only_preprocess(obs):
    obs[:5] = obs[:5]/20.  # Normalizamos números entre [-20, 20], si sale de ahí se asume que puede valer mas que 1 o
                           # menos que -1.
    obs[5:] = obs[5:]/32.27  # 32.27 es el valor máximo de concentración inicial.
    return obs

def bact_only_preprocess(obs):
    obs[:5] = obs[:5]/20.  # Normalizamos números entre [-20, 20], si sale de ahí se asume que puede valer mas que 1 o
                           # menos que -1.
    obs[5:][np.isnan(obs[5:])] = 1e-5
    obs[5:][np.isinf(obs[5:])] = 1e-5
    obs[5:][obs[5:] <= 0.] = 1e-5
    obs[5:] = np.log10(obs[5:])  # Lo pasamos a escala logarítmica
    return obs

def perox_bact_only_norm(obs):
    obs[:10] = obs[:10] / 20.  # Normalizamos números entre [-20, 20], si sale de ahí se asume que puede valer mas que 1 o
                             # menos que -1.

    for i in range(10, 250, 20):
        obs[i:i+20][obs[i:i+20] != -1.] = obs[i:i+20][obs[i:i+20] != -1.] / np.abs(np.max(obs[i:i+20][obs[i:i+20] != -1.]))  # perox  Normalizamos cada curva entre 0 y 1.

    obs[250:][obs[250:] <= 0.] = -1.
    obs[250:][obs[250:] != -1.] = np.log10(obs[250:][obs[250:] != -1.])
    for i in range(250, 490, 20):
        obs[i:i + 20][obs[i:i + 20] != -1.] = obs[i:i + 20][obs[i:i + 20] != -1.] / np.abs(np.max(obs[i:i + 20][obs[i:i + 20] != -1.]))  # perox  Normalizamos cada curva entre 0 y 1.

    return obs

def perox_only_norm(obs):
    obs[:5] = obs[:5] / 20.  # Normalizamos números entre [-20, 20], si sale de ahí se asume que puede valer mas que 1 o
                             # menos que -1.
    obs[5:][obs[5:] <= 0.] = 1e-5
    for i in range(5, 245, 20):
        obs[i:i+20] = obs[i:i+20] / np.abs(np.max(obs[i:i+20]))  # perox  Normalizamos cada curva entre 0 y 1.
    return obs

def perox_only_norm_log(obs):
    positive_index = obs[:5] >= 0  # Extraer parametros positivos
    negative_index = obs[:5] < 0  # Extraer parámetros negativos

    obs[:5][positive_index][obs[:5][positive_index] < 1e-5] = 1e-5  # No permitir valores positivos por debajo de 1e-5

    obs[:5][positive_index] = (np.log10(
        obs[:5][positive_index]) + 10) / 20.  # Convertimos a la escala lineal mediante el

    obs[:5][negative_index][
        obs[:5][negative_index] > -1e-5] = -1e-5  # No permitir valores negativos por encima de -1e-5
    obs[:5][negative_index] = - (np.log10(np.abs(obs[:5][negative_index])) + 10) / 20.  # Convertimos a la escala lineal
    # mediante el logaritmo y
    # normalizamos entre 0 y 1
    obs[5] = np.log10(obs[5])

    obs[5:][obs[5:] <= 0.] = 1e-5

    for i in range(5, 245, 20):
        obs[i:i+20] = obs[i:i+20] / np.abs(np.max(obs[i:i+20]))  # perox  Normalizamos cada curva entre 0 y 1.
    return obs

def bact_only_norm(obs):
    if np.any(np.isinf(obs)) or np.any(np.isnan(obs)):
        print()
    obs[:5] = obs[:5] / 20.  # Normalizamos números entre [-20, 20], si sale de ahí se asume que puede valer mas que 1 o
                             # menos que -1.
    obs[5:][obs[5:] <= 0.] = 1e-5
    obs[5:] = np.log10(obs[5:])  # Lo pasamos a escala logarítmica
    obs[5:][obs[5:] <= 0.] = 1e-5
    for i in range(5, 245, 20):
        obs[i:i + 20] = obs[i:i + 20] / np.abs(np.max(obs[i:i + 20]))  # Normalizamos cada curva entre 0 y 1.
    return obs

def perox_only_norm_hind(obs):
    positive_index = obs[:5] >= 0  # Extraer parametros positivos
    negative_index = obs[:5] < 0  # Extraer parámetros negativos

    obs[:5][positive_index][obs[:5][positive_index] < 1e-5] = 1e-5  # No permitir valores positivos por debajo de 1e-5

    obs[:5][positive_index] = (np.log10(
        obs[:5][positive_index]) + 10) / 20.  # Convertimos a la escala lineal mediante el

    obs[:5][negative_index][
        obs[:5][negative_index] > -1e-5] = -1e-5  # No permitir valores negativos por encima de -1e-5
    obs[:5][negative_index] = - (np.log10(np.abs(obs[:5][negative_index])) + 10) / 20.  # Convertimos a la escala lineal
    # mediante el logaritmo y
    # normalizamos entre 0 y 1
    obs[5] = np.log10(obs[5])

    obs[5:][obs[5:] <= 0.] = 1e-5
    obs[5:] = np.log10(obs[5:])  # Lo pasamos a escala logarítmica
    indices = [(5, 19), (19, 33), (33, 44), (44, 55), (55, 66), (66, 77), (77, 91), (91, 105), (105, 119),
               (119, 133), (133, 147), (147, 161), (161, 175), (175, 189), (189, 200), (200, 211), (211, 222),
               (222, 233), (233, 247), (247, 261), (261, 275), (275, 289), (289, 303), (303, 317)]
    for i in range(24):
        obs[indices[i][0]:indices[i][1]] = \
            obs[indices[i][0]:indices[i][1]] / np.abs(obs[indices[i][0]])  # Normalizamos cada curva entre 0 y 1.
        obs[indices[i][0]:indices[i][1]] = np.clip(obs[indices[i][0]:indices[i][1]], -1., np.max(obs[indices[i][0]:indices[i][1]]))

    return obs

def perox_only_norm_hind_linear(obs):
    obs[:5] = obs[:5] / 20.  # Normalizamos números entre [-20, 20], si sale de ahí se asume que puede valer mas que 1 o
    # menos que -1.
    obs[5:][obs[5:] <= 0.] = 1e-5
    obs[5:] = np.log10(obs[5:])  # Lo pasamos a escala logarítmica

    indices = [(5, 19), (19, 33), (33, 44), (44, 55), (55, 66), (66, 77), (77, 91), (91, 105), (105, 119),
               (119, 133), (133, 147), (147, 161), (161, 175), (175, 189), (189, 200), (200, 211), (211, 222),
               (222, 233), (233, 247), (247, 261), (261, 275), (275, 289), (289, 303), (303, 317)]
    for i in range(24):
        obs[indices[i][0]:indices[i][1]] = \
            obs[indices[i][0]:indices[i][1]] / np.abs(obs[indices[i][0]])  # Normalizamos cada curva entre 0 y 1.
        obs[indices[i][0]:indices[i][1]] = np.clip(obs[indices[i][0]:indices[i][1]], -1., np.max(obs[indices[i][0]:indices[i][1]]))
    return obs

def bact_only_norm_hind(obs):

    positive_index = obs[:4] >= 0  # Extraer parametros positivos
    negative_index = obs[:4] < 0  # Extraer parámetros negativos


    obs[:4][positive_index][obs[:4][positive_index] < 1e-5] = 1e-5  # No permitir valores positivos por debajo de 1e-5

    obs[:4][positive_index] = (np.log10(obs[:4][positive_index]) + 10) / 20.  # Convertimos a la escala lineal mediante el
                                                                        # logaritmo y normalizamos entre 0 y 1

    obs[:4][negative_index][obs[:4][negative_index] > -1e-5] = -1e-5  # No permitir valores negativos por encima de -1e-5
    obs[:4][negative_index] = - (np.log10(np.abs(obs[:4][negative_index])) + 10) / 20.  # Convertimos a la escala lineal
                                                                                      # mediante el logaritmo y
                                                                                      # normalizamos entre 0 y 1
    obs[5] = np.log10(obs[5])

    obs[5:][obs[5:] <= 0.] = 1e-5
    obs[5:] = np.log10(obs[5:])  # Lo pasamos a escala logarítmica
    indices = [(5, 29), (29, 53), (53, 74), (74, 92), (92, 107), (107, 116), (116, 140), (140, 164), (164, 188),
               (188, 212), (212, 236), (236, 260), (260, 284), (284, 308), (308, 329), (329, 347), (347, 362),
               (362, 371), (371, 395), (395, 419), (419, 443), (443, 467), (467, 491), (491, 515)]
    for i in range(24):
        obs[indices[i][0]:indices[i][1]] = \
            obs[indices[i][0]:indices[i][1]] / np.abs(obs[indices[i][0]])  # Normalizamos cada curva entre 0 y 1.
        obs[indices[i][0]:indices[i][1]] = np.clip(obs[indices[i][0]:indices[i][1]], -1., np.max(obs[indices[i][0]:indices[i][1]]))

    return obs

def bact_only_norm_hind_linear(obs):
    obs[:5] = obs[:5] / 20.  # Normalizamos números entre [-20, 20], si sale de ahí se asume que puede valer mas que 1 o
    # menos que -1.
    obs[5:][obs[5:] <= 0.] = 1e-5
    obs[5:] = np.log10(obs[5:])  # Lo pasamos a escala logarítmica

    indices = [(5, 29), (29, 53), (53, 74), (74, 92), (92, 107), (107, 116), (116, 140), (140, 164), (164, 188),
               (188, 212), (212, 236), (236, 260), (260, 284), (284, 308), (308, 329), (329, 347), (347, 362),
               (362, 371), (371, 395), (395, 419), (419, 443), (443, 467), (467, 491), (491, 515)]
    for i in range(24):
        obs[indices[i][0]:indices[i][1]] = \
            obs[indices[i][0]:indices[i][1]] / np.abs(obs[indices[i][0]])  # Normalizamos cada curva entre 0 y 1.
        obs[indices[i][0]:indices[i][1]] = np.clip(obs[indices[i][0]:indices[i][1]], -1., np.max(obs[indices[i][0]:indices[i][1]]))

    return obs
