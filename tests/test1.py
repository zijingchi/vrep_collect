#import h5py
#import keras
#from train.imglike import mdnconstruct
#from keras.models import load_model
from processing.readDataFromFile import DataFromIndex, DataFromDirPkl, DataFromIndexImgLike
import numpy as np
from processing.angle_dis import cal_avo_dir
import os
import pickle
#from train.mdn import sample_from_output


datapath = '/home/czj/Downloads/ur5expert3/'
h5file1 = '../train/h5files/imglike_mdn_cpt5.h5'


def adddim(nparray):
    newa = np.empty(tuple([1]) + nparray.shape, float)
    newa[0] = nparray
    return newa


#model1 = mdnconstruct()
#model1.load_weights(h5file1)

datatest = DataFromIndexImgLike(datapath, rad2deg=True, load_img=False)
with open(os.path.join(datapath, 'list0.pkl'), 'rb') as f:
    lists = pickle.load(f)
    train_list = lists['train']

np.random.shuffle(train_list)
test_list = train_list[:100]

test_ob, test_ac = datatest.read_from_indexes(test_list)
ac_list = []
keys = []
for key, ac in test_ac.items():
    ac_list.append(ac)
    keys.append(key)
ac_list = np.array(ac_list)
pred_aclist = []
'''for key, ob in test_ob.items():
    ob = adddim(np.expand_dims(ob['matrix'], -1))
    pred_ac = model1.predict(ob)
    ac = sample_from_output(pred_ac, 3, 10)
    pred_aclist.append(np.linalg.norm(ac))
pred_aclist = np.array(pred_aclist)'''
print(ac_list.mean(axis=0))
print(ac_list.var(axis=0))
#print(pred_aclist.mean())