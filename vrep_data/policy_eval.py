from vrep_env import vrep_env, vrep
import os
import cv2
import time
import pickle
from keras.models import load_model
from processing.angle_dis import config_dis, obs_pt
from train.training_imgless2 import model_with_1dconv
from vrep_data.vrep_with_policy1 import UR5DaggerSample
import numpy as np

pi = np.pi


class PolicyEval(UR5DaggerSample):

    def __init__(
            self,
            server_addr='127.0.0.1',
            server_port=19997,
            scene_path=None,
            modelweight=None):

        UR5DaggerSample.__init__(
            self,
            server_addr,
            server_port,
            scene_path,
            modelweight
        )
        self.model = model_with_1dconv(5)
        self.model.load_weights(modelweight)

    def reset(self):
        if self.sim_running:
            self.stop_simulation()
        while self.sim_running:
            self.stop_simulation()

        self.target_joint_pos = np.array([0.05 * np.random.randn(), 0.01 * np.random.randn() - pi / 3,
                                          0.05 * np.random.randn() - pi / 3, 0.01 * np.random.randn(),
                                          0.01 * np.random.randn() + pi / 2])

        self.start_simulation()
        colcheck = self.set_obs_pos2()

        self.step_simulation()
        if colcheck == 1:
            self.set_joints(self.init_joint_pos)
            self.step_simulation()

        return colcheck

    def reset2(self):
        if self.sim_running:
            self.stop_simulation()
        while self.sim_running:
            self.stop_simulation()

        self.start_simulation()

        self.obstacle_pos = np.array([0., 0.4, 0.2])
        self.obj_set_position(self.obstable, self.obstacle_pos)
        self.obstacle_ori = np.zeros(3)
        self.obj_set_orientation(self.obstable, self.obstacle_ori)
        self.set_joints(self.target_joint_pos)
        tip_pos = self.obj_get_position(self.tip)
        tip_ori = self.obj_get_orientation(self.tip)
        self.obj_set_position(self.goal_viz, tip_pos)
        self.obj_set_orientation(self.goal_viz, tip_ori)

        self.set_joints(self.init_joint_pos)
        self.step_simulation()
        found = self._ask_vrep()
        return found

    def _ask_vrep(self):
        found = False
        self.set_joints(self.init_joint_pos)
        self.step_simulation()
        clientID = self.cID
        inFloats = self.init_joint_pos + self.target_joint_pos.tolist()
        emptyBuff = bytearray()
        n_path, path, res = self._calPathThroughVrep(clientID, 400, inFloats, emptyBuff)
        if (res == 0) & (n_path != 0):
            np_path = np.array(path)
            re_path = np_path.reshape((n_path, 5))
            thresh = 0.1
            c0 = np.array(self.init_joint_pos)
            final_path = [c0]
            for c in re_path:
                if config_dis(c, c0) > thresh:
                    final_path.append(c)
                    c0 = c
            if c0.any() != np.array(self.target_joint_pos).any():
                final_path.append(np.array(self.target_joint_pos))
            self.n_path = len(final_path)
            self.path = final_path
            # print('obstacle_pos:', self.obstacle_pos)
            found = True
        if res == 3:
            time.sleep(5)
        return found

    def step2(self, t):
        self.set_joints(self.path[t])
        self._make_observation()
        self.step_simulation()

    def step(self, t):
        self._make_observation()  # make an observation
        # predict the action from the model
        config = self.observation['joint'][0:5]
        x1 = self._transpose(config)
        x2 = self._transpose(self.target_joint_pos[0:5])
        x3 = obs_pt(self.obstacle_pos, self.obstacle_ori)
        x3.shape = (1, 8, 3)
        action = self.model.predict([x1, x2, x3])
        # action = action[0]
        action = np.deg2rad(action[0])
        self._make_action(action)  # make the action
        emptyBuff = bytearray()
        colcheck = self._checkInitCollision(self.cID, emptyBuff)
        amp_between = config_dis(self.target_joint_pos[0:5], config)
        check = (amp_between < 0.2) or (colcheck == 1)
        reason = None
        if amp_between < 0.2:
            reason = 'reaching'
        if colcheck == 1:
            reason = 'colliding'
        self.step_simulation()

        return check, reason, amp_between


def main():
    path0 = os.getcwd()
    hi = path0.find('home') + 5
    homepath = path0[:path0.find('/', hi)]
    workpath = homepath + '/vrep_path_dataset/eval3/'
    path1 = path0[:path0.rfind('/')]
    model_path = os.path.join(path1, 'train/h5files/model_big_weights4.h5')

    if not os.path.exists(workpath):
        os.mkdir(workpath)
    dirlist = os.listdir(workpath)
    numlist = [int(s) for s in dirlist]
    if len(numlist) == 0:
        maxdir = -1
    else:
        maxdir = max(numlist)
    os.chdir(workpath)
    env = PolicyEval(modelweight=model_path)
    nci = 0
    si = 0
    asi = 0
    for i in range(maxdir+1, maxdir+80):
        print('iter:', i)
        collision = env.reset()
        if collision:
            # obstacle presence
            nci = nci + 1
            if not os.path.exists(os.path.join(workpath, str(i))):
                os.mkdir(str(i))
                os.mkdir(str(i) + '/pre')
                os.mkdir(str(i) + '/pre/img1')
                os.mkdir(str(i) + '/pre/img2')
            for t in range(60):
                check, reason, amp_between = env.step(t)
                if check:
                    print(reason)
                    if reason == 'reaching':
                        si = si + 1
                    elif reason == 'colliding':
                        print(amp_between)
                    break
                cv2.imwrite(str(i) + '/pre/img1/' + str(t) + '.jpg', env.observation['image1'])
                cv2.imwrite(str(i) + '/pre/img2/' + str(t) + '.jpg', env.observation['image2'])
            # obstacle absent
            os.mkdir(str(i) + '/abs')
            os.mkdir(str(i) + '/abs/img1')
            os.mkdir(str(i) + '/abs/img2')
            found = env.reset2()
            if found:
                for t in range(60):
                    if t > env.n_path - 1:
                        break
                    env.step2(t)
                    cv2.imwrite(str(i) + '/abs/img1/' + str(t) + '.jpg', env.observation['image1'])
                    cv2.imwrite(str(i) + '/abs/img2/' + str(t) + '.jpg', env.observation['image2'])
        else:
            print("collision at initial or target pose")
    print('success rate:', nci, si, si/nci)

    env.close()


if __name__ == '__main__':
    main()
