from vrep_env import vrep_env, vrep
import os
import cv2
import time
import pickle
from train.train_mdn1 import model_with_latentspace_mdn
from processing.angle_dis import obs_pt2, config_dis
from vrep_data.collect_from_vrep1 import UR5WithCameraSample
from processing.fknodes import tipcoor
from train.mdn import *
import numpy as np

pi = np.pi


class MDN_policy(UR5WithCameraSample):
    """
    mdn policy execution class
    """
    metadata = {'render.modes': [], }

    def __init__(
            self,
            server_addr='127.0.0.1',
            server_port=19997,
            scene_path=None,
            modelweights=None,
            askvrep=True
    ):

        UR5WithCameraSample.__init__(
            self,
            server_addr,
            server_port,
            scene_path,
        )

        self.model = model_with_latentspace_mdn(3)
        self.model.load_weights(modelweights)
        self.askvrep = askvrep
        print('UR5VrepEnv: initialized')

    def _cal_depth(self, chandle, zfar, znear):
        depmat = self.obj_get_depth_matrix(chandle)
        depmat = znear * np.ones(np.shape(depmat)) + (zfar - znear) * depmat
        return depmat

    def _transpose(self, nparray):
        #m = np.mat(nparray).T
        nparray.shape = (1, nparray.size)
        return nparray

    def step(self, t):
        self._make_observation()  # make an observation
        # predict the action from the model
        config = self.observation['joint']
        thresh = 0.1
        x1 = self._transpose(config[:3])
        x2 = self._transpose(self.target_joint_pos[:3])
        #x3 = obs_pt2(self.obstacle_pos, self.obstacle_ori)
        x3 = self._transpose(self.obstacle_pos)
        x4 = self._transpose(self.obstacle_ori)
        #x3.shape = (1, 8, 3)
        param = self.model.predict([x1, x2, x3, x4])
        action = sample_from_output(param, 3, 15, 1, 0.2)
        action = np.concatenate((action, np.zeros(2)))
        action = np.deg2rad(action)
        action = 1.4*action + thresh * (self.target_joint_pos - config) / np.linalg.norm(self.target_joint_pos[:3] - config[:3])
        #print(action)
        self._make_action(action)  # make the action
        self.step_simulation()

        emptyBuff = bytearray()
        colcheck = self._checkInitCollision(self.cID, emptyBuff)
        amp_between = config_dis(self.target_joint_pos, config)
        check1 = amp_between < thresh
        check2 = colcheck == 1
        if check1:
            print('reaching')
        if check2:
            print('colliding')
        if self.askvrep:
            inFloats = config.tolist() + self.target_joint_pos.tolist()
            minConfigs = int(300 * np.linalg.norm(self.target_joint_pos - config))

            n_path, path, res = self._calPathThroughVrep(self.cID, minConfigs, inFloats, emptyBuff)
            expert_action = []
            if (res == 0) & (n_path != 0):
                np_path = np.array(path)
                re_path = np_path.reshape((n_path, 5))
                for p in re_path:
                    n = config_dis(p, config)
                    if n > thresh:
                        expert_action = p - config
                        break
            elif res == 3:
                print('timeout')
                time.sleep(6)
            else:
                print('no')
            check3 = len(expert_action) == 0
            check = check1 or check2 or check3

            if check3:
                print('expert action not found')
            # else:
            #    self._make_action(expert_action)
            self.step_simulation()

            return action, expert_action, check
        else:
            check = (amp_between < thresh) or (colcheck == 1)
            return action, None, check

    def set_obs_pos2(self):
        self.set_joints(self.init_joint_pos)
        tip_pos = self.obj_get_position(self.tip)
        alpha = np.random.rand()
        self.obstacle_pos = alpha * np.array(tip_pos) + (1 - alpha) * tipcoor(self.target_joint_pos.tolist() + [0])
        self.obstacle_pos[0] = self.obstacle_pos[0] + 0.05 * np.random.randn()
        self.obstacle_pos[1] = self.obstacle_pos[1] + 0.05 * np.random.randn()
        self.obstacle_pos[2] = self.obstacle_pos[2] + 0.2 * (np.random.rand() + 0.5)
        self.obj_set_position(self.obstable, self.obstacle_pos)
        self.obstacle_ori = 0.1 * np.random.rand(3)
        self.obstacle_ori[2] = self.obstacle_ori[2] + pi / 2
        self.obj_set_orientation(self.obstable, self.obstacle_ori)
        emptyBuff = bytearray()
        colcheck1 = self._checkInitCollision(self.cID, emptyBuff)

        self.set_joints(self.target_joint_pos)
        tip_pos = self.obj_get_position(self.tip)
        tip_ori = self.obj_get_orientation(self.tip)
        tip_obs_col = (tip_pos[2] < self.obstacle_pos[2] + 0.15) & (tip_pos[1] <
                                                                    self.obstacle_pos[1] + 0.04) & (
                                  tip_pos[1] > self.obstacle_pos[1] - 0.04) & (tip_pos[0] <
                                                                               self.obstacle_pos[0] + 0.27) & (
                                  tip_pos[0] > self.obstacle_pos[0] - 0.27)
        self.obj_set_position(self.goal_viz, tip_pos)
        self.obj_set_orientation(self.goal_viz, tip_ori)
        colcheck2 = self._checkInitCollision(self.cID, emptyBuff)
        if (colcheck1 == 0) & (colcheck2 == 0) & (not tip_obs_col):
            return 1
        else:
            return 0

    def reset(self):
        if self.sim_running:
            self.stop_simulation()
        while self.sim_running:
            self.stop_simulation()

        self.target_joint_pos = np.array([0.2 * np.random.randn(), 0.1 * np.random.randn() - pi / 3,
                                          0.2 * np.random.randn() - pi / 3, 0.3 * np.random.randn(),
                                          0.2 * np.random.randn() + pi / 2])

        self.start_simulation()
        colcheck = self.set_obs_pos2()

        self.inits = {'target_joint_pos': self.target_joint_pos,
                      'obstacle_pos': np.array(self.obstacle_pos),
                      'obstacle_ori': self.obstacle_ori}
        self.step_simulation()
        if colcheck == 1:
            self.current_state = self.init_joint_pos
            self.set_joints(self.init_joint_pos)
            self.step_simulation()

        return colcheck

    def render(self, mode='human', close=False):
        pass

    def calAngDis(self, angles, targetangles):
        return np.linalg.norm(angles - targetangles)


def main(args):
    path0 = os.getcwd()
    hi = path0.find('home') + 5
    homepath = path0[:path0.find('/', hi)]
    workpath = homepath + '/vdp/mdn5/'
    path1 = path0[:path0.rfind('/')]
    model_path = os.path.join(path1, 'train/h5files/3dof_latent_mdn_weights_11.h5')
    if not os.path.exists(workpath):
        os.mkdir(workpath)
    dirlist = os.listdir(workpath)
    numlist = [int(s) for s in dirlist if os.path.isdir(os.path.join(workpath, s))]
    if len(numlist) == 0:
        maxdir = -1
    else:
        maxdir = max(numlist)
    os.chdir(workpath)
    askvrep = False
    env = MDN_policy(modelweights=model_path,
                     askvrep=askvrep)
    i = maxdir + 1
    while i < maxdir + 200:
        print('iter:', i)
        collision = env.reset()
        if collision:
            if not os.path.exists(os.path.join(workpath, str(i))):
                os.mkdir(str(i))
                os.mkdir(str(i) + '/img1')
                os.mkdir(str(i) + '/img2')
            obs = []
            acs = []
            exp_acs = []
            for t in range(80):
                action, expert_action, check = env.step(t)
                if check:
                    break
                obs.append(env.observation['joint'])
                cv2.imwrite(str(i) + '/img1/' + str(t) + '.jpg', env.observation['image1'])
                cv2.imwrite(str(i) + '/img2/' + str(t) + '.jpg', env.observation['image2'])
                acs.append(action)
                if askvrep:
                    exp_acs.append(expert_action)
                env.current_state = env.observation['joint']
            if askvrep:
                data = {'inits': env.inits, 'observations': obs, 'actions': exp_acs, 'policy': acs}
            else:
                data = {'inits': env.inits, 'observations': obs, 'actions': acs}
            if len(obs) != 0:
                with open(str(i) + '/data.pkl', 'wb') as f:
                    pickle.dump(data, f)
            i = i + 1
        else:
            print("collision at initial or target pose")
    # print("Episode finished after {} timesteps.\tTotal reward: {}".format(t+1,total_reward))
    env.close()
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))
