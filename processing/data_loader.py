import os
import numpy as np
import pickle
import re


class PathPlanDset(object):
    def __init__(self, expert_path, train_fraction=0.7, traj_limitation=-1, randomize=True):
        #self.datafromindex = DataFromIndex(expert_path, rad2deg=False, load_img=False)
        all_files = [f for f in os.listdir(expert_path) if f.isdigit() and os.path.exists(os.path.join(expert_path, f, 'data.pkl'))]
        all_files.sort(key=lambda s: int(s[0]))
        self.obs_list = []
        self.acs_list = []
        for f in all_files:
            with open(os.path.join(expert_path, f, 'data.pkl'), 'rb') as pf:
                data = pickle.load(pf)
                inits = data['inits']
                acs = data['actions']
                obs = data['observations']
                obstacle_pos = inits['obstacle_pos']
                #obstacle_ori = inits['obstacle_ori']
                for t in range(1, len(obs)):
                    inp = obs[t]
                    inp = np.append(inp, inits['target_joint_pos'])
                    inp = np.append(inp, obstacle_pos)
                    #inp = np.append(inp, obstacle_ori)
                    inp = np.append(inp, tipcoor(obs[t])[3:-3])
                    inp = np.append(inp, tipcoor(inits['target_joint_pos'])[3:-3])
                    self.obs_list.append(inp)
                    self.acs_list.append(acs[t])

        if traj_limitation > 0:
            self.obs_list = self.obs_list[:traj_limitation]
            self.acs_list = self.acs_list[:traj_limitation]
        self.obs_list = np.array(self.obs_list)
        self.acs_list = np.rad2deg(np.array(self.acs_list))
        train_size = int(train_fraction*len(self.obs_list))
        self.dset = VDset(self.obs_list, self.acs_list, randomize)
        # for behavior cloning
        '''self.train_set = VDset(self.obs_list[:train_size, :],
                               self.acs_list[:train_size, :],
                               randomize)
        self.val_set = VDset(self.obs_list[train_size:, :],
                             self.acs_list[train_size:, :],
                             randomize)
        self.pointer = 0
        self.train_pointer = 0
        self.test_pointer = 0'''

    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError


class VDset(object):
    def __init__(self, inputs, labels, randomize):
        self.inputs = inputs
        self.labels = labels
        assert len(self.inputs) == len(self.labels)
        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx]
            self.labels = self.labels[idx]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels


def ur5fk(thetas):
    d1 = 8.92e-2
    d2 = 0.11
    d5 = 9.475e-2
    d6 = 1.1495e-1
    a2 = 4.251e-1
    a3 = 3.9215e-1
    d0 = 0.3
    All = np.zeros((6, 4, 4))
    All[:, 3, 3] = 1
    for i in range(5):
        All[i, 0, 0] = np.cos(thetas[i])
        All[i, 0, 1] = -np.sin(thetas[i])
    All[0, 1, 0] = np.sin(thetas[0])
    All[0, 1, 1] = np.cos(thetas[0])
    All[0, 2, 3] = d1
    All[0, 2, 2] = 1

    All[1, 2, 0] = np.sin(thetas[1])
    All[1, 2, 1] = np.cos(thetas[1])
    All[1, 1, 2] = -1
    All[1, 1, 3] = -d2

    All[2, 1, 0] = np.sin(thetas[2])
    All[2, 1, 1] = np.cos(thetas[2])
    All[2, 0, 3] = a2
    All[2, 2, 2] = 1

    All[3, 1, 0] = np.sin(thetas[3])
    All[3, 1, 1] = np.cos(thetas[3])
    All[3, 0, 3] = a3
    All[3, 2, 2] = 1

    All[4, 2, 0] = np.sin(thetas[4])
    All[4, 2, 1] = np.cos(thetas[4])
    All[4, 1, 3] = -d5
    All[4, 1, 2] = -1

    All[5, :, :] = np.eye(4)
    All[5, 1, 3] = -d6

    A0 = np.zeros((4, 4))
    A0[0, 1] = 1
    A0[1, 0] = -1
    A0[2, 2] = 1
    A0[3, 3] = 1
    A0[2, 3] = d0
    return All, A0


def tipcoor(thetas):
    pi = np.pi
    thetas_0 = np.array([0, pi / 2, 0, pi / 2, pi])
    thetas = thetas + thetas_0
    All, A0 = ur5fk(thetas)
    ps = []
    for A in All:
        A0 = A0 @ A
        ps.extend(A0[:3, 3])
    return np.array(ps)


class ExpertDataset(object):
    def __init__(self, expert_path, train_fraction=0.7, listpkl='list0.pkl', traj_limitation=-1, load_img=False):
        self.path = expert_path
        self.load_img = load_img
        self.listpkl = listpkl
        self.traj_limit = traj_limitation
        self.split_train(train_fraction)
        self.n_train = len(self.train_list)
        self.n_val = len(self.vali_list)
        self.train_pointer = 0
        self.val_pointer = 0

    def split_train(self, fraction):
        if os.path.exists(os.path.join(self.path, self.listpkl)):
            with open(os.path.join(self.path, self.listpkl), 'rb') as f:
                data = pickle.load(f)
                self.train_list = data['train']
                self.vali_list = data['test']
        else:
            #self.listpkl = 'list0.pkl'
            dirlist = os.listdir(self.path)
            id_list = []
            np.random.shuffle(dirlist)
            for d in dirlist:
                subdir = os.path.join(self.path, d)
                if os.path.isdir(subdir):
                    datapkl = os.path.join(subdir, 'data.pkl')
                    if os.path.exists(datapkl):
                        with open(datapkl, 'rb') as dataf:
                            data = pickle.load(dataf)
                            '''a0 = data['actions'][0]
                            at = data['actions'][-1]
                            if np.linalg.norm(a0-at)<0.02 and np.random.rand()<0.8:
                                #shutil.rmtree(subdir)
                                continue'''
                            for i in range(len(data['actions'])):
                                id_list.append(d + '-' + str(i))
            if self.traj_limit>0:
                id_list = id_list[:self.traj_limit]
            id_size = len(id_list)
            train_size = int(fraction * id_size)
            # np.random.shuffle(id_list)
            self.train_list = id_list[:train_size]
            self.vali_list = id_list[train_size:]

            np.random.shuffle(self.train_list)
            np.random.shuffle(self.vali_list)

            with open(os.path.join(self.path, self.listpkl), 'wb') as f1:
                pickle.dump({'train': self.train_list, 'test': self.vali_list}, f1)

    def read_single_index(self, index):
        reindex = re.split('\W', index)
        dirname = reindex[0]
        datadir = os.path.join(self.path, dirname)

        t = int(reindex[1])
        pkl_path = os.path.join(datadir, 'data.pkl')
        img1_path = os.path.join(datadir, 'img1/' + reindex[1] + '.jpg')
        img2_path = os.path.join(datadir, 'img2/' + reindex[1] + '.jpg')
        observation = {}
        '''if self.load_img:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            observation['img1'] = img1
            observation['img2'] = img2'''
        with open(pkl_path, 'rb') as pkl_file:
            pkl_data = pickle.load(pkl_file)
            inits = pkl_data['inits']
            obs = pkl_data['observations']
            # config = obs[t]['joint']

            config = obs[t]
            tar_pos = inits['target_joint_pos']
            obstacle_pos = inits['obstacle_pos']
            xyzs = tipcoor(config)[3:-3]
            obs_final = np.concatenate((config, tar_pos, obstacle_pos, xyzs))

            actions = pkl_data['actions']
            action = actions[t]
            if np.linalg.norm(action) > 0.5:
                action = actions[-1]

        observation['config'] = obs_final
        return observation, action

    def read_indexes(self, indexes):
        obs = []
        acs = []
        for index in indexes:
            ob, ac = self.read_single_index(index)
            obs.append(ob['config'])
            acs.append(ac)
        return np.array(obs), np.array(acs)

    def get_next_batch(self, batch_size, train=True):
        if train:
            cur_list = self.train_list
            cur_pointer = self.train_pointer
            cur_len = self.n_train
        else:
            cur_list = self.vali_list
            cur_pointer = self.val_pointer
            cur_len = self.n_val

        #cur_len = len(cur_list)
        end = cur_pointer + batch_size
        if batch_size<0:
            return self.read_indexes(cur_list)
        if end<cur_len:
            if train:
                self.train_pointer = end
            else:
                self.val_pointer = end
            return self.read_indexes(cur_list[cur_pointer:end])
        else:
            if train:
                self.train_pointer = end - cur_len
            else:
                self.val_pointer = end - cur_len
            return self.read_indexes(cur_list[cur_pointer:]+cur_list[:end-cur_len])


import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D


class DataVizer(object):
    def __init__(self, datapath):
        self.path = datapath
        self.subdirs = []

    def load_all_sub(self):
        subdirs = os.listdir(self.path)
        for d in subdirs:
            fullpath = os.path.join(self.path, d)
            if os.path.isdir(fullpath) and os.path.exists(os.path.join(fullpath, 'data.pkl')):
                self.subdirs.append(fullpath)

    def load_traj(self, i):
        pklpath = os.path.join(self.subdirs[i], 'data.pkl')
        with open(pklpath, 'rb') as f:
            data = pickle.load(f)
        traj = data['observations']
        goal = data['inits']['target_joint_pos']
        traj = np.append(traj, np.expand_dims(goal, 0), axis=0)
        return traj

    def viz(self, traj):
        xs, ys, zs = traj[:, 0], traj[:, 1], traj[:, 2]
        fig1 = plt.figure(1)
        ax = Axes3D(fig1)
        ax.scatter(xs, ys, zs, c='b', s=5)
        plt.show()

    def interp(self, traj):
        a1 = traj[:, 0]
        a2 = traj[:, 1]
        a3 = traj[:, 2]
        a4 = traj[:, 3]
        a5 = traj[:, 4]
        tck, u = interpolate.splprep([a1,a2,a3,a4,a5], s=0.1, k=5)
        #x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
        u_fine = np.linspace(0,1,200)
        x_fine, y_fine, z_fine, _, _ = interpolate.splev(u_fine, tck)
        fig2 = plt.figure(2)
        ax = Axes3D(fig2)
        #ax.scatter(a1, a2, a3, c='b', s=5)
        ax.plot(a1, a2, a3, 'b')
        ax.plot(x_fine, y_fine, z_fine, 'r')

        plt.show()


def main():
    dv = DataVizer('/home/czj/Downloads/ur5expert')
    dv.load_all_sub()
    for i in range(40, 50):
        traj = dv.load_traj(i)
        dv.interp(traj)


if __name__ == '__main__':
    main()
