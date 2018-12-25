import h5py
# import pickle
import numpy as np
import os
from processing.readDataFromFile import DataFromDir

datapath1 = '/home/czj/vrep_path_dataset/1/'


def files2h5(datapath, delete):
    for test in os.listdir(datapath):
        if (os.path.isdir(datapath + test)) & (int(test) < 1):
            cur_data = DataFromDir(datapath, test)
            cur_data.read_actions()
            cur_data.read_configs()
            cur_data.read_imgs()
            cur_data.rad2deg()
            hf = h5py.File(os.path.join(cur_data.dir, "data.hdf5"), "w")
            ob = hf.create_group("observation")
            ob["configs"] = cur_data.configs
            ob["img1"] = cur_data.img1
            ob["img2"] = cur_data.img2
            hf["action"] = cur_data.actions
            if delete:
                os.removedirs(cur_data.img1dir)
                os.removedirs(cur_data.img2dir)
                os.unlink(cur_data.acfile)
                os.unlink(cur_data.obsfile)


def deleteh5(basepath):
    pass

def openh5(h5path):
    hf = h5py.File(h5path, "r")
    a1 = hf["action/actions"]
    print(a1.attrs)

def main():
    files2h5(datapath1, False)
    #openh5(datapath1 + "1/data.hdf5")


if __name__ == '__main__':
    main()