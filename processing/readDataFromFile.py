import os
import re
import cv2
import numpy as np
import pickle


class ReadDataBase(object):

    def __init__(self, dataset_path, data_num):
        self.path = dataset_path
        self.img1 = []
        self.img2 = []
        self.actions = []
        self.configs = []

    def subdir_filename(self, sub):
        dir = os.path.join(self.path, sub)
        imgpath = [os.path.join(dir, 'img1'), os.path.join(dir, 'img2')]
        pklname = os.path.join(dir, 'data.pkl')
        return dir, imgpath, pklname

    def strs2float(self, line):
        numbers = re.split("\s+", line)
        numbers = numbers[1:-1]
        float_number = [float(n) for n in numbers]
        return float_number

    def rad2deg(self):
        self.configs = [np.rad2deg(config) for config in self.configs]
        self.actions = [np.rad2deg(action) for action in self.actions]

    def select_single(self, i):
        observation = {"config": self.configs[i], "img1": self.img1[i], "img2": self.img2[i]}
        action = self.actions[i]
        return observation, action

    def load(self, subdir, rad2deg=False):
        pass


class DataFromPkl(ReadDataBase):

    def __init__(self, dataset_path, data_num):
        ReadDataBase.__init__(self, dataset_path, data_num)
        self.inits = []
        self.depth = []

    def load(self, rad2deg=False, load_depth=False):
        pkl_path = os.path.join(self.dir, "data.pkl")
        res = 1
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            self.actions = data['actions']
            self.inits = data['inits']
            obs = data['observations']
            self.img1 = [ob['image1'] for ob in obs]
            self.img2 = [ob['image2'] for ob in obs]
            self.configs = [ob['joint'] for ob in obs]
            if load_depth:
                self.depth = [ob['depth'] for ob in obs]
        if not (len(self.actions) == len(self.configs) == len(self.img1) == len(self.img2)):
            print(self.num, "tuple length not match!")
            self.actions = []
            self.configs = []
            self.img1 = []
            self.img2 = []
            res = 0
        if rad2deg:
            self.rad2deg()
        return res

    def select_single(self, i):
        observation = {"config": self.configs[i], "img1": self.img1[i], "img2": self.img2[i],
                       "depth": self.depth[i], "inits": self.inits}
        action = self.actions[i]
        return observation, action


class DataFromDirPkl(object):

    def __init__(self, dataset_path):
        self.path = dataset_path

    def read_imgs(self, imgpath):
        self.img1 = []
        self.img2 = []
        imgpath1, imgpath2 = imgpath[0], imgpath[1]
        for s in os.listdir(imgpath1):
            imgname = os.path.join(imgpath1, s)
            img = cv2.imread(imgname)
            self.img1.append(img)
        for s in os.listdir(imgpath2):
            imgname = os.path.join(imgpath2, s)
            img = cv2.imread(imgname)
            self.img2.append(img)

    def subdir_filename(self, sub):
        dir = os.path.join(self.path, sub)
        imgpath = [os.path.join(dir, 'img1'), os.path.join(dir, 'img2')]
        pklname = os.path.join(dir, 'data.pkl')
        return dir, imgpath, pklname

    def load(self, subdir, load_img=False):
        _, imgpath, pkl_path = self.subdir_filename(subdir)
        if load_img:
            self.read_imgs(imgpath)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.actions = data['actions']
        self.configs = data['observations']
        ni = len(self.actions)
        if np.linalg.norm(self.actions[0]) > 0.3:
            ni = ni - 1
            self.actions = self.actions[1:]
            self.configs = self.configs[1:]
        self.actions = np.array(self.actions)
        inits = data['inits']
        self.tar_pos = np.empty((ni, 5), dtype=float)
        self.obstacle_pos = np.empty((ni, 3), dtype=float)
        self.obstacle_ori = np.empty((ni, 3), dtype=float)
        for i in range(ni):
            self.tar_pos[i, ] = inits['target_joint_pos']
            self.obstacle_pos[i, ] = inits['obstacle_pos']
            self.obstacle_ori[i, ] = inits['obstacle_ori']

        if not (ni == len(self.configs)):
            print(subdir, "tuple length not match!")
            self.actions = []
            self.configs = []
            ni = 0

        return ni

    def configs_sequence(self, maxsize, n, padding_order='pre'):
        dof = self.configs[0].size
        zlist = self.sequence_maxsize(maxsize, n)
        configs = np.zeros((n, maxsize, dof))
        #lens = []
        for i in range(n):
            for j, s in enumerate(zlist[i]):
                if padding_order == 'post':
                    configs[i, j, ] = self.configs[s]
                elif padding_order == 'pre':
                    configs[i, maxsize-1-j,] = self.configs[s]
        return configs

    def sequence_maxsize(self, maxsize, n):
        """
        maxsize: max length of a sequence
        n: length of the original sequence
        """
        zlist = []
        for i in range(1, min(n, maxsize)+1):
            zlist.append(list(range(i)))
        if n > maxsize:
            for i in range(1, n - maxsize + 1):
                zlist.append(list(range(i, i + maxsize)))
        return zlist


class DataPack(object):

    def __init__(self, path):
        self.path = path
        # self.all_dir = [os.path.join(path, di) for di in os.listdir(path)]
        self.all_dir = os.listdir(path)

    def _pack_select(self, idlist):
        select_observations = {}
        select_actions = {}
        for sud in idlist:
            singledata = DataFromDirPkl(self.path, sud)
            res = singledata.load()
            if res == 0:
                continue
            for i in range(len(singledata.configs)):
                observation, action = singledata.select_single(i)
                id = sud + "-" + str(i)
                select_observations[id] = observation
                select_actions[id] = action
        return select_observations, select_actions


def anglize(vector3d):
    vector3d = vector3d/np.linalg.norm(vector3d)
    phi = np.arccos(vector3d[2])
    theta = np.arctan2(vector3d[1]/np.sin(phi), vector3d[0]/np.sin(phi))
    return np.array([theta, phi])


def deanglize(vector2d):
    return np.array([np.sin(vector2d[1])*np.cos(vector2d[0]),
                     np.sin(vector2d[1])*np.sin(vector2d[0]),
                     np.cos(vector2d[1])])


class DataFromIndex(object):

    def __init__(self, path, rad2deg=False, load_depth=False, load_img=False):
        self.path = path
        # self.indexes = indexes
        # self.reindexes = [re.split('\W', i) for i in indexes]
        self.rad2deg = rad2deg
        self.load_depth = load_depth
        self.load_img = load_img
        # self.selected_obs = {}
        # self.selected_act = {}

    def load_all_indexes(self):
        subdirs = os.listdir(self.path)
        allindexes = []
        for d in subdirs:
            pd = os.path.join(self.path, d, 'data.pkl')
            if os.path.exists(pd):
                with open(pd, 'rb') as f:
                    data = pickle.load(f)
                    allindexes.extend([d+'-'+str(i) for i in range(len(data['actions']))])
        return allindexes

    def read_per_index(self, index):
        reindex = re.split('\W', index)
        dirname = reindex[0]
        datadir = os.path.join(self.path, dirname)
        flagd = (dirname[0] == 'd')
        flagd = False
        t = int(reindex[1])
        pkl_path = os.path.join(datadir, 'data.pkl')
        img1_path = os.path.join(datadir, 'img1/' + reindex[1] + '.jpg')
        img2_path = os.path.join(datadir, 'img2/' + reindex[1] + '.jpg')
        observation = {}
        action = []
        if self.load_img:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            observation['img1'] = img1
            observation['img2'] = img2
        with open(pkl_path, 'rb') as pkl_file:
            pkl_data = pickle.load(pkl_file)
            inits = pkl_data['inits']
            obs = pkl_data['observations']
            # config = obs[t]['joint']

            config = obs[t]
            tar_pos = inits['target_joint_pos']
            if self.load_depth:
                observation['depth'] = obs[t]['depth']
            obstacle_pos = inits['obstacle_pos']
            #obstacle_ori = inits['obstacle_ori']
            xyzs = tipcoor(config)[3:-3]
            obs_final = np.concatenate((config, tar_pos, obstacle_pos, xyzs))
            if flagd:
                dagger_pkl = os.path.join(datadir, 'dagger.pkl')
                with open(dagger_pkl, 'rb') as dagger_file:
                    dagger_data = pickle.load(dagger_file)
                    action = dagger_data['actions'][t]
            else:
                actions = pkl_data['actions']
                action = actions[t]
                if np.linalg.norm(action) > 0.5:
                    action = actions[-1]
                    config = obs[t+1]-action

        #action = anglize(action[:3])
        #action = cal_avo_dir(action, tar_pos, config, 0.1, 5)
        #action = action / 0.04
        #newa = []
        #for i in range(3):
        #    newa.append(min(max(round(action[i]), -3), 3))
        #newa = np.clip(np.round(action), -3, 3)

        if self.rad2deg:
            #config = np.rad2deg(config)
            #tar_pos = np.rad2deg(tar_pos)
            #obstacle_ori = np.rad2deg(obstacle_ori)
            action = np.rad2deg(action)

        observation['config'] = obs_final
        #observation['tar_pos'] = tar_pos
        #observation['obstacle_pos'] = obstacle_pos
        #observation['obstacle_ori'] = obstacle_ori
        return observation, action

    def read_from_indexes(self, indexes):
        selected_obs = {}
        selected_act = {}
        for index in indexes:
            obs, act = self.read_per_index(index)
            selected_obs[index] = obs
            selected_act[index] = act
        return selected_obs, selected_act


from processing.fknodes import tipcoor
from processing.angle_dis import cal_avo_dir


class DataFromIndexImgLike(DataFromIndex):

    def __init__(self, path, rad2deg=False, load_depth=False, load_img=False):
        super(DataFromIndexImgLike, self).__init__(path, rad2deg, load_depth, load_img)


    def read_per_index(self, index):
        reindex = re.split('\W', index)
        dirname = reindex[0]
        datadir = os.path.join(self.path, dirname)
        t = int(reindex[1])
        pkl_path = os.path.join(datadir, 'data.pkl')
        img1_path = os.path.join(datadir, 'img1/' + reindex[1] + '.jpg')
        img2_path = os.path.join(datadir, 'img2/' + reindex[1] + '.jpg')
        observation = {}
        action = []
        if self.load_img:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            observation['img1'] = img1
            observation['img2'] = img2
        with open(pkl_path, 'rb') as pkl_file:
            pkl_data = pickle.load(pkl_file)
            inits = pkl_data['inits']
            obs = pkl_data['observations']
            # config = obs[t]['joint']
            config = obs[t]
            tar_pos = inits['target_joint_pos']
            if self.load_depth:
                observation['depth'] = obs[t]['depth']
            obstacle_pos = inits['obstacle_pos']
            matrix = np.zeros((5, 5))
            matrix[0, :] = config
            matrix[1, :] = tar_pos
            matrix[2:, 0] = obstacle_pos
            xyzs = tipcoor(config)[3:-3]
            matrix[2:, 1:] = xyzs.reshape((-1, 3)).T
            actions = pkl_data['actions']
            action = actions[t]
            action = cal_avo_dir(action, tar_pos, config, 0.1, 3)
            action = action / 0.05
            newa = 0
            for i in range(3):
                newa += (min(2, max(-2, round(action[i]))) + 2) * 5 ** (2 - i)
        if self.rad2deg:
            action = np.rad2deg(action)

        observation['matrix'] = matrix
        return observation, newa

    def read_from_indexes(self, indexes):
        selected_obs = {}
        selected_act = {}
        for index in indexes:
            obs, act = self.read_per_index(index)
            selected_obs[index] = obs
            selected_act[index] = act
        return selected_obs, selected_act