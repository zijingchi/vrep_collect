import pickle
from processing.angle_dis import config_dis
import os
import numpy as np

datapath = '/home/czj/vrep_path_dataset/4_3/'
os.chdir(datapath)
dirs = os.listdir(datapath)
ss = []
for d in dirs:
    pklpath = os.path.join(d, 'data.pkl')
    if os.path.isdir(d) and os.path.exists(pklpath):
        with open(pklpath, 'rb') as f:
            data = pickle.load(f)
            dis = config_dis(data['actions'][3], np.zeros(5))
            ss.append(dis)

ss = np.array(ss)
mean = np.mean(ss)
var = np.var(ss)
print(mean, var)
