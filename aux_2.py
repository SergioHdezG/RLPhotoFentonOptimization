from src.IRL.utils.callbacks import load_expert_memories
import numpy as np
import pandas as pd
import cv2
import glob
import os

# exp_dir = "expert_demonstrations/"
# exp_name = 'human_expert_carla_wheel'
# exp_memory = load_expert_memories(exp_dir, exp_name, load_action=True)
#
# exp_name = 'human_expert_carla_wheel_2'
# exp_memory_1 = load_expert_memories(exp_dir, exp_name, load_action=True)
#
# exp_name = 'human_expert_carla_wheel_5500'
# exp_memory_2 = load_expert_memories(exp_dir, exp_name, load_action=True)
#
# exp_name = 'human_expert_carla_3'
# exp_memory_3 = load_expert_memories(exp_dir, exp_name, load_action=True)
#
# exp_name = 'human_expert_carla_4'
# exp_memory_4 = load_expert_memories(exp_dir, exp_name, load_action=True)
#
# exp_name = 'human_expert_carla_5'
# exp_memory_5 = load_expert_memories(exp_dir, exp_name, load_action=True)
#
# exp_name = 'human_expert_carla_6'
# exp_memory_6 = load_expert_memories(exp_dir, exp_name, load_action=True)
#
# exp_memory = np.concatenate((exp_memory_1, exp_memory_2, exp_memory_3, exp_memory_4, exp_memory_5, exp_memory_6), axis=0)

# exp_memory = np.concatenate((exp_memory, exp_memory_1), axis=0)
#
# exp_memory = pd.DataFrame(exp_memory, columns=['obs', 'action', 'reward', 'done'])
# #
# exp_name = 'human_expert_carla_wheel_5500'
# exp_memory.to_pickle(exp_dir+exp_name+'.pkl'
# myfiles = []
# dirFiles = os.listdir('/home/serch/TFM/IRL2/CAPORL/environments/examples/_out/')
# nojpeg = []
# for files in dirFiles: #filter out all non jpgs
#     if '.jpeg' in files:
#         myfiles.append(files)
#     else:
#         nojpeg.append(files)
#
# # sorting by name
# myfiles.sort(key=lambda f: int(f[-10:-5]))
#
# rgb_files = []
# rgb_files_aux = []
# seg_files = []
# seg_files_aux = []
# dep_files = []
# dep_files_aux = []
#
# # select type pof img
# for f in myfiles:
#     if 'rgb' in f:
#         rgb_files.append(f)
#     elif 'seg' in f:
#         seg_files.append(f)
#     elif 'dep' in f:
#         dep_files.append(f)
#
# rgb_num = [int(f[-10:-5])+1 for f in rgb_files]
# seg_num = [int(f[-10:-5]) for f in seg_files]
# dep_num = [int(f[-10:-5]) for f in dep_files]
#
# for f in rgb_files:
#     if int(f[-10:-5])+1 in seg_num and int(f[-10:-5])+1 in dep_num:
#         rgb_files_aux.append(f)
# rgb_files = rgb_files_aux
# rgb_num = [int(f[-10:-5])+1 for f in rgb_files]
# for f in seg_files:
#     if int(f[-10:-5]) in rgb_num and int(f[-10:-5]) in dep_num:
#         seg_files_aux.append(f)
# seg_files = seg_files_aux
# seg_num = [int(f[-10:-5]) for f in seg_files]
# for f in dep_files:
#     if int(f[-10:-5]) in seg_num and int(f[-10:-5]) in rgb_num:
#         dep_files_aux.append(f)
# dep_files = dep_files_aux
# dep_num = [int(f[-10:-5]) for f in dep_files]
#
# url = '/home/serch/TFM/IRL2/CAPORL/environments/examples/_out/'
#
# rgb_images =[]
# seg_images = []
# dep_images = []

# for rgb, seg, dep in zip(rgb_files, seg_files, dep_files):
#     rgb = cv2.imread(url + rgb)
#     seg = cv2.imread(url + seg)
#     dep = cv2.imread(url + dep)
#     rgb_images.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
#     seg_images.append(cv2.cvtColor(seg, cv2.COLOR_BGR2RGB))
#     dep_images.append(cv2.cvtColor(dep, cv2.COLOR_BGR2RGB))
#
# pasos = len(rgb_images)
# for i in range(pasos):
#     rgb = np.flip(rgb_images[i], 1)
#     rgb_images.append(rgb)
#     seg = np.flip(seg_images[i], 1)
#     seg_images.append(seg)
#     dep = np.flip(dep_images[i], 1)
#     dep_images.append(dep)
#
# np.save('rgb_street.npy', rgb_images)
# np.save('seg_street.npy', seg_images)
# np.save('dep_street.npy', dep_images)

# rgb_images_1 = np.load('rgb.npy')
# seg_images_1 = np.load('seg.npy')
# dep_images_1 = np.load('dep.npy')

# rgb_images = np.load('/home/serch/TFM/IRL2/CAPORL/environments/carla/rgb_street.npy')
# seg_images = np.load('/home/serch/TFM/IRL2/CAPORL/environments/carla/seg_street.npy')
# dep_images = np.load('/home/serch/TFM/IRL2/CAPORL/environments/carla/dep_street.npy')

# rgb_images = np.concatenate([rgb_images, rgb_images_1])
# seg_images = np.concatenate([seg_images, seg_images_1])
# dep_images = np.concatenate([dep_images, dep_images_1])
# np.save('rgb_carla.npy', rgb_images)
# np.save('seg_carla.npy', seg_images)
# np.save('dep_carla.npy', dep_images)

# for rgb, seg, dep in zip(rgb_images, seg_images, dep_images):
#     cv2.imshow('rgb', rgb)
#     # cv2.imshow('seg', seg)
#     # cv2.imshow('dep', dep)
#     cv2.waitKey(1)

test_images = np.load('/home/serch/TFM/IRL3/saved_models/Carla/DGX/test_vid_99.npy')
for rgb in test_images:
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow('rgb', rgb)
    # cv2.imshow('seg', seg)
    # cv2.imshow('dep', dep)
    cv2.waitKey(1)

print('done')

