#import h5py
#import keras
from keras.models import load_model
from processing.readDataFromFile import DataFromIndex, DataFromDirPkl
import numpy as np
from processing.angle_dis import cal_avo_dir


datapath = '/home/ubuntu/vdp/4_7/'
h5file1 = '../train/h5files/3dof_model.h5'


def adddim(nparray):
    newa = np.empty(tuple([1]) + nparray.shape, float)
    newa[0] = nparray
    return newa


model1 = load_model(h5file1)

idlist = ['27-5', '48-29']

datatest = DataFromIndex(datapath, rad2deg=False, load_img=False)
data2 = DataFromDirPkl(datapath)
ni = data2.load('800')
configs = data2.configs
tar = data2.tar_pos[0]
actions = data2.actions
for i in range(ni):
    inputs = [adddim(configs[i][:3]),
              adddim(tar[:3]),
              adddim(data2.obstacle_pos[0]),
              adddim(data2.obstacle_ori[0])]
    #inputs = [configs[i][:3], tar[:3], data2.obstacle_pos[0], data2.obstacle_ori[0]]
    a_pred = model1.predict(inputs)
    a_true = cal_avo_dir(actions[i], tar, configs[i], 0.1, 3)
    a_true = np.rad2deg(a_true)

    a_err = a_pred[0] - a_true
    print('a_true', np.linalg.norm(a_true))
    print('a_pred', a_pred[0])
#obs, act = datatest.read_from_indexes(idlist)

