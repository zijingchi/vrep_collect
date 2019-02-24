
import os
import time
import pickle
from vrep_data.collect_from_vrep1 import UR5WithCameraSample
import numpy as np

pi = np.pi


class UR5ObsColCheck(UR5WithCameraSample):

    def __init__(self, n,
            server_addr='127.0.0.1',
            server_port=19997,
            scene_path=None,
    ):

        super().__init__(
            server_addr,
            server_port,
            scene_path)
        self.n = n
        self.target_joint_pos = np.array([0.2 * np.random.randn(), 0.1 * np.random.randn() - pi / 3,
                                          0.2 * np.random.randn() - pi / 3, 0.3 * np.random.randn(),
                                          0.2 * np.random.randn() + pi / 2])

        theta1_left = -1.
        theta1_right = 1.
        theta2_left = -1.2
        theta2_right = 0.4
        theta3_left = -2.7
        theta3_right = 0.
        theta4_left = -1.
        theta4_right = 1.
        theta5_left = 0.3
        theta5_right = 2.7

        theta1_sample = np.linspace(theta1_left, theta1_right, 20)
        theta2_sample = np.linspace(theta2_left, theta2_right, 20)
        theta3_sample = np.linspace(theta3_left, theta3_right, 20)
        theta4_sample = np.linspace(theta4_left, theta4_right, 3)
        theta5_sample = np.linspace(theta5_left, theta5_right, 3)

        thetas_sample = []
        alpha = 1.2
        for t1 in theta1_sample:
            for t2 in theta2_sample:
                for t3 in theta3_sample:
                    if (alpha*t2 + t3 < -pi) or (alpha*t2 + t3 > -pi/4):
                        continue
                    for t4 in theta4_sample:
                        for t5 in theta5_sample:
                            thetas_sample.append([t1, t2, t3, t4, t5])
        print(len(thetas_sample))
        self.thetas_sample = np.array(thetas_sample)
        print('theta samples initialized')

    def reset(self):
        n = self.n
        obs_poses = np.empty(shape=(n, 6), dtype=float)
        for i in range(n):
            obs_pos = [0.4 * np.random.randn(), -0.49 + 0.3 * np.random.randn(), 0.32 + 0.3 * np.random.randn()]
            obs_poses[i, :3] = np.array(obs_pos)
            obs_ori = 0.2 * np.random.randn(3)
            obs_ori[2] = obs_ori[2] + pi / 2
            obs_poses[i, 3:] = obs_ori

        if self.sim_running:
            self.stop_simulation()
        while self.sim_running:
            self.stop_simulation()

        self.obstacle = obs_poses
        self.start_simulation()

    def check_col_states(self, obs):

        emptybuff = bytearray()
        n_states = len(self.thetas_sample)
        col_states = np.zeros(n_states)

        for i in range(n_states):
            theta = self.thetas_sample[i]
            self.obj_set_position(self.obstable, obs[:3])
            self.obj_set_orientation(self.obstable, obs[3:])
            self.set_joints(theta)
            #self.step_simulation()
            colcheck = self._checkInitCollision(self.cID, emptybuff)
            col_states[i] = colcheck

        return col_states


def examine_states(datapath):
    dirs = os.listdir(datapath)
    os.chdir(datapath)
    col_states = []
    for fname in dirs:
        if 'col_states' in fname:
            with open(fname, 'rb') as f:
                col_state = pickle.load(f)
                col_states.append(col_state)
    col_states = np.array(col_states)
    col_counts = np.sum(col_states, axis=0)
    r1 = []
    r2 = []
    r3 = []
    r4 = []
    for i in range(len(col_counts)):
        if col_counts[i]<5:
            r1.append(i)
        elif col_counts[i]<10:
            r2.append(i)
        elif col_counts[i]<20:
            r3.append(i)
        else:
            r4.append(i)
    col_sort = np.argsort(col_counts)
    return col_states, col_counts, col_sort



def main():
    path0 = os.getcwd()
    hi = path0.find('home') + 5
    homepath = path0[:path0.find('/', hi)]
    workpath = homepath + '/vdp/colstates'
    datapath1 = workpath + '/2'
    examine_states(datapath1)
    if not os.path.exists(workpath):
        os.mkdir(workpath)
    dirlist = os.listdir(workpath)
    numlist = [int(s) for s in dirlist if os.path.isdir(os.path.join(workpath, s))]
    if len(numlist) == 0:
        maxdir = -1
    else:
        maxdir = max(numlist)
    os.chdir(workpath)
    next_dir = str(maxdir+1)
    os.mkdir(next_dir)

    env = UR5ObsColCheck(50)
    env.reset()
    for i in range(len(env.obstacle)):
        print(i)
        start_time = time.time()
        col_states_per_obs = env.check_col_states(env.obstacle[i])
        col_states_pkl = str(i) + 'col_states.pkl'
        with open(os.path.join(next_dir, col_states_pkl), 'wb') as f2:
            pickle.dump(col_states_per_obs, f2)
        end_time = time.time()
        print('cost %d s' % (end_time - start_time))

    states_pkl = 'states.pkl'
    obs_pkl = 'obs.pkl'
    with open(os.path.join(next_dir, states_pkl), 'wb') as f1:
        pickle.dump(env.thetas_sample, f1)
    with open(os.path.join(next_dir, obs_pkl), 'wb') as f3:
        pickle.dump(env.obstacle, f3)


if __name__ == '__main__':
    main()
