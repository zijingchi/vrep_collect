import os
import re
import cv2
import numpy as np
import pickle


class ReadDataBase(object):

    def __init__(self, dataset_path, data_num):
        self.num = data_num
        self.path = dataset_path
        self.img1 = []
        self.img2 = []
        self.actions = []
        self.configs = []
        self.dir = os.path.join(dataset_path, data_num)

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

    def load(self):
        pass


class DataFromDir(ReadDataBase):

    def __init__(self, dataset_path, data_num):
        ReadDataBase.__init__(self, dataset_path, data_num)
        self.img1dir = os.path.join(self.dir , "img1")
        self.img2dir = os.path.join(self.dir, "img2")
        self.acfile = os.path.join(self.dir, "action.txt")
        self.obsfile = os.path.join(self.dir, "observation_joints.txt")

    def read_imgs(self):
        img1 = []
        img2 = []
        for name in os.listdir(self.img1dir):
            img_name = os.path.join(self.img1dir, name)
            img = cv2.imread(img_name)
            img1.append([int(name[:-4]), cv2.cvtColor(img, cv2.COLOR_BGR2RGB)])
        for name in os.listdir(self.img2dir):
            img_name = os.path.join(self.img2dir, name)
            img = cv2.imread(img_name)
            img2.append([int(name[:-4]), cv2.cvtColor(img, cv2.COLOR_BGR2RGB)])
        img1.sort(key=lambda s: s[0])
        img2.sort(key=lambda s: s[0])
        self.img1 = [a[1] for a in img1]
        self.img2 = [a[1] for a in img2]

    def read_actions(self):
        with open(self.acfile, "r") as f:
            actions = f.readlines()
            for i in range(len(actions)):
                faction = self.strs2float(actions[i])
                if len(faction) == 6:
                    self.actions.append(faction)
                elif len(faction) == 4:
                    nextaction = actions[i+1]
                    nextfaction = self.strs2float(nextaction)
                    faction.extend(nextfaction)
                    self.actions.append(faction)

    def read_configs(self):
        fconfigs = []
        with open(self.obsfile, "r") as f:
            configs = f.readlines()
            for i in range(len(configs)):
                config = configs[i]
                fconfig = self.strs2float(config)
                if len(fconfig) == 6:
                    fconfigs.append(fconfig)
                elif len(fconfig) == 4:
                    nextconfig = configs[i+1]
                    nextfconfig = self.strs2float(nextconfig)
                    fconfig.extend(nextfconfig)
                    fconfigs.append(fconfig)
        self.configs = fconfigs

    def load(self, rad2deg=True):
        res = 1
        self.read_imgs()
        self.read_configs()
        self.read_actions()
        if rad2deg:
            self.rad2deg()
        if not (len(self.actions) == len(self.configs) == len(self.img1) == len(self.img2)):
            print(self.num, "tuple length not match!")
            self.actions = []
            self.configs = []
            self.img1 = []
            self.img2 = []
            res = 0
        return res


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


class DataFromDirPkl(DataFromDir):

    def __init__(self, dataset_path, data_num):
        DataFromDir.__init__(self, dataset_path, data_num)
        self.inits = []
        self.depth = []

    def load(self, rad2deg=False, load_depth=False):
        pkl_path = os.path.join(self.dir, "data.pkl")
        res = 1
        self.read_imgs()
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            self.actions = data['actions']
            self.inits = data['inits']
            obs = data['observations']
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
        observation = {"config": self.configs[i], "img1": self.img1[i], "img2": self.img2[i], "inits": self.inits}
        if len(self.depth) != 0:
            observation["depth"] = self.depth[i]
        action = self.actions[i]
        return observation, action


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
            tar_pos = np.array(inits['target_joint_pos'])
            if self.load_depth:
                observation['depth'] = obs[t]['depth']
            obstacle_pos = np.array(inits['obstacle_pos'])
            obstacle_ori = inits['obstacle_ori']
            if flagd:
                dagger_pkl = os.path.join(datadir, 'dagger.pkl')
                with open(dagger_pkl, 'rb') as dagger_file:
                    dagger_data = pickle.load(dagger_file)
                    action = dagger_data['actions'][t]
            else:
                actions = pkl_data['actions']
                action = actions[t]

        if self.rad2deg:
            #config = np.rad2deg(config)
            #tar_pos = np.rad2deg(tar_pos)
            #obstacle_ori = np.rad2deg(obstacle_ori)
            action = np.rad2deg(action)

        observation['config'] = config
        observation['tar_pos'] = tar_pos
        observation['obstacle_pos'] = obstacle_pos
        observation['obstacle_ori'] = obstacle_ori
        return observation, action

    def read_from_indexes(self, indexes):
        selected_obs = {}
        selected_act = {}
        for index in indexes:
            obs, act = self.read_per_index(index)
            selected_obs[index] = obs
            selected_act[index] = act
        return selected_obs, selected_act
