import pickle
import numpy as np
import os
import re
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class AnalyzeCol(object):

    def __init__(self, upperdir):
        self.upperdir = upperdir
        allfiles = os.listdir(upperdir)
        col_files = []
        obs_files = None
        states_files = None
        for file in allfiles:
            if 'col_states' in file:
                col_files.append(file)
            elif file == 'states.pkl':
                states_files = file
            elif file == 'obs.pkl':
                obs_files = file

        if not (obs_files and states_files):
            raise FileExistsError('file not exist')

        self.files = col_files
        os.chdir(upperdir)

    def load_pkl(self, fname):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        return data

    def load_col_states(self, fname):
        match = re.compile('\d*')
        order = int(match.findall(fname)[0])
        data = self.load_pkl(fname)
        col_states_i = np.nonzero(data)[0]
        return col_states_i, order

    def organize_col(self):
        states = self.load_pkl('states.pkl')
        obs = self.load_pkl('obs.pkl')
        all_col_states = []
        for fname in self.files:
            col_states_i, i = self.load_col_states(fname)
            all_col_states.append([states[col_states_i], obs[i]])

        self.all_col_states = all_col_states


def main():
    datapath = '/home/ubuntu/vdp/colstates/0'
    ac = AnalyzeCol(datapath)
    col_state1, i1 = ac.load_col_states(ac.files[37])
    states = ac.load_pkl('states.pkl')
    cs = states[col_state1]
    fig1 = plt.figure(1)
    ax = Axes3D(fig1)
    ax.scatter(cs[:, 0], cs[:, 1], cs[:, 2])
    ax.set_xlabel('theta_1')
    ax.set_xbound(-1,1)
    ax.set_ylabel('theta_2')
    ax.set_ybound(-1.2,0.4)
    ax.set_zlabel('theta_3')
    ax.set_zbound(-2.7,0)
    '''fig = plt.figure()
    plt.scatter(states[:,1],states[:,2])'''
    plt.show()


if __name__ == '__main__':
    main()
