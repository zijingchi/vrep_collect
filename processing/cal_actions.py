import os
import pickle
# import math
from processing.readDataFromFile import DataFromDirPkl, DataFromIndex
import numpy as np

datapath = '/home/czj/vrep_path_dataset/4/'


class PreProcess(DataFromDirPkl):

    def __init__(self, datapath, num):
        DataFromDirPkl.__init__(self, datapath, num)
        self.load(rad2deg=True)

    def cal_amp(self, actions):
        return [np.linalg.norm(np.array(a)) for a in actions]

    def cal_amp_between(self, config1, config2):
        return np.linalg.norm(np.array(config1) - np.array(config2))

    def _select_configs(self, thresh):
        configs = self.configs
        selected = [0]
        i1 = 0
        for i in range(len(configs)):
            a1 = configs[i1]
            a2 = configs[i]
            amp = self.cal_amp_between(a1, a2)
            if amp > thresh:
                i1 = i
                selected.append(i)
        if i1 != len(configs)-1:
            selected.append(len(configs) - 1)
        return selected

    def return_list(self, thresh):
        selected = self._select_configs(thresh)
        pklfile = os.path.join(self.dir, 'seleted_list.pkl')
        with open(pklfile, 'wb') as f:
            pickle.dump(selected, f)
        """newdata = {}
        actions = [self.actions[i] for i in selected]
        configs = [self.configs[i] for i in selected]
        img1s = [self.img1[i] for i in selected]
        img2s = [self.img2[i] for i in selected]"""
        return selected


def cal_mean(ds):
    sum = 0
    for d in ds:
        sum = sum + d
    return sum/len(ds)


def main():
    dirlist = os.listdir(datapath)
    id_list = []
    for d in dirlist:
        data = PreProcess(datapath, d)
        data.return_list(4.0)
        #for i in range(50):
        #    id_list.append(d + '-' + str(i))
    #id_size = len(id_list)
    """train_size = int(0.1 * id_size)
    np.random.shuffle(id_list)
    train_list = id_list[:train_size]
    datafromindex = DataFromIndex(datapath, rad2deg=True, load_img=False)
    sobs, sacts = datafromindex._read_from_indexes(train_list)
    sum = 0
    for key, act in sacts.items():
        normact = np.linalg.norm(act)
        sum = sum + normact
    mean = sum/train_size
    print(mean)"""


main()
