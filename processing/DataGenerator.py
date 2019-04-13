import numpy as np
import keras
# import os
from processing.readDataFromFile import DataFromIndex, DataFromDirPkl
from processing.angle_dis import obs_pt2, cal_avo_dir
from keras.preprocessing.sequence import pad_sequences


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
        self.datafromindex = DataFromIndex(datapath, rad2deg=False, load_img=False)
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
        dof = self.data_size
        configs = np.empty((self.batch_size, 5), dtype=float)
        actions = np.empty((self.batch_size, dof), dtype=float)
        tar_pos_config = np.empty((self.batch_size, 5), dtype=float)
        #obstacle = np.empty((self.batch_size, 8, 3), dtype=float)
        obstacle_pos = np.empty((self.batch_size, 3), dtype=float)
        obstacle_ori = np.empty((self.batch_size, 3), dtype=float)

        for i, ID in enumerate(list_IDs_temp):
            obs, act = self.datafromindex.read_per_index(ID)
            configs[i,] = obs['config'][:5]
            tar_pos_config[i,] = obs['tar_pos'][:5]
            obstacle_pos[i,] = obs['obstacle_pos']
            obstacle_ori[i,] = obs['obstacle_ori']
            avo = cal_avo_dir(act, obs['tar_pos'], obs['config'], 0.1, dof)
            actions[i,] = np.rad2deg(avo)

        return [configs, tar_pos_config, obstacle_pos, obstacle_ori], actions


class CustomDataGenWthTarCfgSqc(keras.utils.Sequence):

    def __init__(self, datapath, list_IDs, dof, max_step, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.dof = dof
        self.maxstep = max_step
        self.datapath = datapath
        self.dataloader = DataFromDirPkl(datapath)
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
        dof = self.dof
        actions = np.empty((0, dof), float)
        tar_joint_pos = np.empty((0, 5), float)
        obs_pos = np.empty((0, 3), float)
        obs_ori = np.empty((0, 3), float)
        configs_sequence = np.empty((0, self.maxstep, 5), float)
        for i, ID in enumerate(list_IDs_temp):
            ni = self.dataloader.load(ID)
            configs = self.dataloader.configs_sequence(self.maxstep, ni, padding_order='post')
            acs = self.dataloader.actions
            #tar = self.dataloader.tar_pos[0]
            '''for j in range(ni):
                acs[j] = cal_avo_dir(acs[j], tar, self.dataloader.configs[j], 0.1, 5)'''
            actions = np.append(actions, acs, 0)
            tar_joint_pos = np.append(tar_joint_pos, self.dataloader.tar_pos, 0)
            obs_pos = np.append(obs_pos, self.dataloader.obstacle_pos, 0)
            obs_ori = np.append(obs_ori, self.dataloader.obstacle_ori, 0)
            configs_sequence = np.append(configs_sequence, configs, 0)

        return [configs_sequence, tar_joint_pos, obs_pos, obs_ori], actions


def sph_theta_phi(nparray):
    nparray = nparray/np.linalg.norm(nparray)
    phi = np.arccos(nparray[2])
    theta = np.arccos(nparray[0]/np.sin(phi))
    return np.array([theta, phi])
