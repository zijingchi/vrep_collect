import numpy as np
import keras
# import os
from processing.readDataFromFile import DataFromIndex


class CustomDataGenWthImg(keras.utils.Sequence):

    def __init__(self, datapath, list_IDs, data_size, img_dim, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.data_size = data_size
        self.dim = img_dim
        self.datapath = datapath
        self.datafromindex = DataFromIndex(datapath, rad2deg=True)
        self.list_IDs = list_IDs
        self.indexes = np.arange(len(self.list_IDs))
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        configs = np.empty((self.batch_size, 6), dtype=float)
        actions = np.empty((self.batch_size, 6), dtype=float)
        img1s = np.empty((self.batch_size, *self.dim))
        img2s = np.empty((self.batch_size, *self.dim))

        for i, ID in enumerate(list_IDs_temp):
            obs, act = self.datafromindex.read_per_index(ID)
            configs[i, ] = obs['config']
            img1s[i, ] = obs['img1']
            img2s[i, ] = obs['img2']
            actions[i, ] = act

        return [configs, img1s, img2s], actions


class CustomDataGenWthTarCfg(keras.utils.Sequence):

    def __init__(self, datapath, list_IDs, data_size, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.data_size = data_size
        self.datapath = datapath
        self.datafromindex = DataFromIndex(datapath, rad2deg=True, load_img=False)
        self.list_IDs = list_IDs
        self.indexes = np.arange(len(self.list_IDs))
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        configs = np.empty((self.batch_size, 5), dtype=float)
        actions = np.empty((self.batch_size, 5), dtype=float)
        tar_pos_config = np.empty((self.batch_size, 5), dtype=float)
        obstacle_posnori = np.empty((self.batch_size, 6), dtype=float)
        #obstacle_pos = np.empty((self.batch_size, 3), dtype=float)
        #obstacle_ori = np.empty((self.batch_size, 3), dtype=float)

        for i, ID in enumerate(list_IDs_temp):
            obs, act = self.datafromindex.read_per_index(ID)
            configs[i,] = obs['config']
            tar_pos_config[i,] = obs['tar_pos']
            obstacle_posnori[i,] = np.concatenate((obs['obstacle_pos'], obs['obstacle_ori']))
            #obstacle_ori[i,] = obs['obstacle_ori']
            actions[i,] = act

        return [configs, tar_pos_config, obstacle_posnori], actions
