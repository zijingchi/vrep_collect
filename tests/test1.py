import h5py
#import keras
from keras.models import load_model
from processing.readDataFromFile import DataFromIndex
import numpy as np
import os

datapath = '/home/czj/vrep_path_dataset/4/'
h5file1 = '../train/h5files/model3_14.h5'
h5file2 = '../train/h5files/model3_11.h5'
modelFile1 = h5py.File(h5file1, 'r')
modelFile2 = h5py.File(h5file2, 'r')
#model_weights = modelFile1['model_weights']

model1 = load_model(h5file1)
model2 = load_model(h5file2)
# idlist = ['27-5', '48-29', '103-15', '143-34', '177-9', '267-40', '300-3', '321-47']
dirlist = os.listdir(datapath)
id_list = []
for d in dirlist:
    for i in range(50):
        id_list.append(d + '-' + str(i))
id_size = len(id_list)
test_size = int(0.2 * id_size)
np.random.shuffle(id_list)
testlist = id_list[:test_size]
datatest = DataFromIndex('/home/czj/vrep_path_dataset/4/', rad2deg=True, load_img=False)
obs, act = datatest._read_from_indexes(testlist)

configs = np.empty((test_size, 6), dtype=float)
actions = np.empty((test_size, 6), dtype=float)
tar_pos_config = np.empty((test_size, 6), dtype=float)
obstacle_posnori = np.empty((test_size, 6), dtype=float)
for i in range(test_size):
    id = testlist[i]
    ac = act[id]
    ob = obs[id]
    configs[i,] = ob['config']
    tar_pos_config[i,] = ob['tar_pos']
    obstacle_posnori[i,] = np.concatenate((ob['obstacle_pos'], ob['obstacle_ori']))
    actions[i,] = ac

loss1, accuracy1 = model1.evaluate([configs, tar_pos_config, obstacle_posnori], actions)
print(accuracy1)
loss2, accuracy2 = model2.evaluate([configs, tar_pos_config, obstacle_posnori], actions)
print(accuracy2)
#print(configs[0,])


def transpose(nparray):
    nparray.shape = (nparray.size, 1)
    return np.transpose(nparray)
