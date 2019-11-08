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


def thresh_planes(theta2_left,theta2_right,theta3_left,theta3_right,low,high):
    theta2_sample = np.linspace(theta2_left, theta2_right, 32)
    theta3_sample = np.linspace(theta3_left, theta3_right, 32)
    lin_low = {}
    lin_high = {}
    for t2 in theta2_sample:
        t3low = theta3_left - 1
        t3high = theta3_right + 1
        for t3 in theta3_sample:
            z = 1.2*t2 + t3
            if z<low:
                t3low = t3
            elif z>high:
                t3high = t3
                break
        if t3low != theta3_left-1:
            lin_low[t2]=t3low
        if t3high != theta3_right+1:
            lin_high[t2]=t3high
    return lin_low, lin_high


def draw_boundaries(theta1_left,theta1_right,theta2_left,theta2_right,theta3_left,theta3_right,low,high):
    lin_low, lin_high = thresh_planes(theta2_left, theta2_right, theta3_left, theta3_right, low, high)
    theta2_sample = np.linspace(theta2_left, theta2_right, 32)
    lowplane_t2 = []
    highplane_t2 = []
    for t2 in theta2_sample:
        if t2 in lin_low.keys():
            lowplane_t2.append(t2)
        if t2 in lin_high.keys():
            highplane_t2.append(t2)
    lowplane_t2 = np.array(lowplane_t2)
    highplane_t2 = np.array(highplane_t2)
    low_n, high_n = lowplane_t2.size, highplane_t2.size
    theta1_sample = np.linspace(theta1_left, theta1_right, 32)
    lowx = np.zeros(32 * low_n)
    lowy = np.zeros(32 * low_n)
    lowz = np.zeros(32 * low_n)
    for i in range(low_n):
        lowx[i * 32:i * 32 + 32] = theta1_sample
        lowy[i * 32:i * 32 + 32] = lowplane_t2[i] * np.ones(32)
        lowz[i * 32:i * 32 + 32] = lin_low[lowplane_t2[i]] * np.ones(32)
    highx = np.zeros(32 * high_n)
    highy = np.zeros(32 * high_n)
    highz = np.zeros(32 * high_n)
    for i in range(high_n):
        highx[i * 32:i * 32 + 32] = theta1_sample
        highy[i * 32:i * 32 + 32] = highplane_t2[i] * np.ones(32)
        highz[i * 32:i * 32 + 32] = lin_high[highplane_t2[i]] * np.ones(32)
    return lowx,lowy,lowz,highx,highy,highz


def main():
    datapath = '/home/czj/col_check6'
    ac = AnalyzeCol(datapath)
    col_state1, i1 = ac.load_col_states(ac.files[40])
    states = ac.load_pkl('states.pkl')
    cs = states[col_state1]
    fig1 = plt.figure(1)
    ax = Axes3D(fig1)
    ax.scatter(cs[:, 0], cs[:, 1], cs[:, 2], s=20, alpha=0.5, c=cs[:, 0]*cs[:, 2])
    ax.set_xlabel('theta_1')
    theta1_left, theta1_right = -1.6, 1.6
    ax.set_xbound(theta1_left, theta1_right)
    ax.set_ylabel('theta_2')
    theta2_left, theta2_right = -1.9, 0.9
    ax.set_ybound(theta2_left, theta2_right)
    ax.set_zlabel('theta_3')
    theta3_left, theta3_right = -2.7, 0.3
    ax.set_zbound(theta3_left,theta3_right)
    lowx, lowy, lowz, highx, highy, highz = draw_boundaries(theta1_left, theta1_right,
                                                            theta2_left, theta2_right,
                                                            theta3_left, theta3_right,
                                                            -6*np.pi/5, 0)
    ax.scatter(lowx, lowy, lowz, c='b', s=10, alpha=0.2)
    ax.scatter(highx, highy, highz, c='r', s=10, alpha=0.2)
    path1 = '/home/czj/vrep_path_dataset/4_3/101'
    with open(path1+'/data.pkl','rb') as f1:
        data = pickle.load(f1)
        target = data['inits']['target_joint_pos']
        init = data['observations'][0]
    ax.scatter(init[0],init[1],init[2],s=20,alpha=1,c='g')
    ax.scatter(target[0],target[1],target[2],s=20,alpha=1,c='r')
    '''fig = plt.figure()
    plt.scatter(states[:,1],states[:,2])'''
    plt.show()


if __name__ == '__main__':
    main()
