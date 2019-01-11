from vrep_env import vrep_env, vrep
import os
import cv2
import time
import pickle
from keras.models import load_model
from processing.angle_dis import config_dis, obs_pt
from train.training_imgless2 import model_with_config_n_target2
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
        self.model = model_with_config_n_target2(5, 1)
        self.model.load_weights(modelweight)

    def reset(self):
        if self.sim_running:
            self.stop_simulation()
        while self.sim_running:
            self.stop_simulation()

        self.target_joint_pos = np.array([0.2 * np.random.randn(), 0.1 * np.random.randn() - pi / 4,
                                          0.2 * np.random.randn() - pi / 3, 0.3 * np.random.randn(),
                                          0.2 * np.random.randn() + pi / 2])

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
        self.set_joints(self.init_joint_pos)
        self.obstacle_pos = np.array([0., 0.2, 0.4])
        self.obj_set_position(self.obstable, self.obstacle_pos)
        self.obstacle_ori = np.zeros(3)
        self.obj_set_orientation(self.obstable, self.obstacle_ori)
        self.set_joints(self.target_joint_pos)
        tip_pos = self.obj_get_position(self.tip)
        tip_ori = self.obj_get_orientation(self.tip)
        self.obj_set_position(self.goal_viz, tip_pos)
        self.obj_set_orientation(self.goal_viz, tip_ori)
        self.step_simulation()

    def step(self, t):
        self._make_observation()  # make an observation
        # predict the action from the model
        config = self.observation['joint'][0:5]
        x1 = self._transpose(np.rad2deg(config))
        x2 = self._transpose(np.rad2deg(self.target_joint_pos[0:5]))
        x3 = obs_pt(self.obstacle_pos, self.obstacle_ori)
        action = self.model.predict([x1, x2, x3])
        # action = action[0]
        action = np.deg2rad(action[0])
        self._make_action(action)  # make the action
        emptyBuff = bytearray()
        colcheck = self._checkInitCollision(self.cID, emptyBuff)
        amp_between = np.linalg.norm(self.target_joint_pos[0:5] - config)
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
    workpath = homepath + '/vrep_path_dataset/eval2/'
    path1 = path0[:path0.rfind('/')]
    model_path = os.path.join(path1, 'train/h5files/model10_17_weights.h5')

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
    for i in range(maxdir+1, maxdir+100):
        print('iter:', i)
        collision = env.reset()
        if collision:
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
            os.mkdir(str(i) + '/abs')
            os.mkdir(str(i) + '/abs/img1')
            os.mkdir(str(i) + '/abs/img2')
            env.reset2()
            for t in range(60):
                check, reason, amp_between = env.step(t)
                if check:
                    print(reason)
                    if reason == 'reaching':
                        asi = si + 1
                    elif reason == 'colliding':
                        print(amp_between)
                    break
                cv2.imwrite(str(i) + '/abs/img1/' + str(t) + '.jpg', env.observation['image1'])
                cv2.imwrite(str(i) + '/abs/img2/' + str(t) + '.jpg', env.observation['image2'])
        else:
            print("collision at initial or target pose")
    print('success rate:', nci, si, si/nci)

    env.close()


if __name__ == '__main__':
    main()
