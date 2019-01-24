from vrep_env import vrep_env, vrep
import os
import cv2
import time
import pickle
from keras.models import load_model
from train.training_imgless2 import model_with_1dconv
from processing.angle_dis import obs_pt2, config_dis
from processing.fknodes import tipcoor
from vrep_data.collect_from_vrep1 import UR5WithCameraSample
import numpy as np

pi = np.pi


class UR5DaggerSample(UR5WithCameraSample):
    metadata = {'render.modes': [], }

    def __init__(
            self,
            server_addr='127.0.0.1',
            server_port=19997,
            scene_path=None,
            modelfile=None
    ):

        UR5WithCameraSample.__init__(
            self,
            server_addr,
            server_port,
            scene_path,
        )

        self.model = None
        if modelfile:
            self.model = model_with_1dconv(5)
            self.model.load_weights(modelfile)

        print('UR5DaggerSample: initialized')

    def set_obs_pos2(self):
        self.set_joints(self.init_joint_pos)
        tip_pos = self.obj_get_position(self.tip)
        alpha = np.random.rand()+0.2
        self.obstacle_pos = alpha * np.array(tip_pos) + (1 - alpha) * tipcoor(self.target_joint_pos.tolist() + [0])
        self.obstacle_pos[0] = self.obstacle_pos[0] + 0.05 * np.random.randn()
        self.obstacle_pos[1] = self.obstacle_pos[1] + 0.05 * np.random.randn()
        self.obstacle_pos[2] = self.obstacle_pos[2] + 0.3 * (np.random.rand() + 0.3)
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
                       self.obstacle_pos[1] + 0.04) & (tip_pos[1] > self.obstacle_pos[1] - 0.04) & (tip_pos[0] <
                       self.obstacle_pos[0] + 0.27) & (tip_pos[0] > self.obstacle_pos[0] - 0.27)
        self.obj_set_position(self.goal_viz, tip_pos)
        self.obj_set_orientation(self.goal_viz, tip_ori)
        colcheck2 = self._checkInitCollision(self.cID, emptyBuff)
        if (colcheck1 == 0) & (colcheck2 == 0) & (not tip_obs_col):
            return 1
        else:
            return 0

    def _transpose(self, nparray):
        m = np.mat(nparray).T
        m.shape = (1, nparray.size)
        return m

    def step(self, t):
        self._make_observation()    # make an observation
        action = np.zeros(5)
        # predict the action from the model
        config = self.observation['joint']
        if self.model:
            x1 = self._transpose(config)
            x2 = self._transpose(self.target_joint_pos)
            #x3 = self._transpose(np.concatenate((self.obstacle_pos, self.obstacle_ori)))
            x3 = obs_pt2(self.obstacle_pos, self.obstacle_ori)
            x3.shape = (1, 8, 3)
            action = self.model.predict([x1, x2, x3])
            #action = action[0]
            action = np.deg2rad(action[0])
            self._make_action(action)   # make the action

        # ask the expert to give the right action if exists (here the expert is the ompl algorithm used in v-rep
        inFloats = config.tolist() + self.target_joint_pos.tolist()
        minConfigs = int(200 * np.linalg.norm(self.target_joint_pos - config))
        emptyBuff = bytearray()
        n_path, path, res = self._calPathThroughVrep(self.cID, minConfigs, inFloats, emptyBuff)
        thresh = 0.4
        expert_action = []
        if (res == 0) & (n_path != 0):
            np_path = np.array(path)
            re_path = np_path.reshape((n_path, 5))
            for p in re_path:
                n = config_dis(p, config)
                if n > thresh:
                    expert_action = p - config
                    #print(n)
                    break
        if res == 3:
            print('timeout')
            time.sleep(6)
        colcheck = self._checkInitCollision(self.cID, emptyBuff)
        amp_between = config_dis(self.target_joint_pos, config)

        check = (amp_between < thresh) or (colcheck == 1) or (expert_action == [])
        if check:
            if amp_between < thresh:
                print('reaching')
            if colcheck == 1:
                print('colliding')
            if expert_action == []:
                print('expert action not found')
        #else:
        #    self._make_action(expert_action)

        self.step_simulation()

        return action, expert_action, check

    def reset(self):
        if self.sim_running:
            self.stop_simulation()
        while self.sim_running:
            self.stop_simulation()

        self.target_joint_pos = np.array([0.2*np.random.randn(), 0.1*np.random.randn()-pi/3,
        0.2*np.random.randn()-pi/3, 0.3*np.random.randn(), 0.2*np.random.randn()+pi/2])
        
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


def main(args):
    path0 = os.getcwd()
    hi = path0.find('home') + 5
    homepath = path0[:path0.find('/', hi)]
    workpath = homepath+'/vdp/4_2/'
    path1 = path0[:path0.rfind('/')]
    model_path = os.path.join(path1, 'train/h5files/model_conv1d_weights1.h5')
    if not os.path.exists(workpath):
        os.mkdir(workpath)
    dirlist = os.listdir(workpath)
    numlist = [int(s) for s in dirlist if os.path.isdir(os.path.join(workpath, s))]
    if len(numlist) == 0:
        maxdir = -1
    else:
        maxdir = max(numlist)
    os.chdir(workpath)
    env = UR5DaggerSample(modelfile=None)
    for i in range(maxdir+1, maxdir+50):
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
            for t in range(40):
                action, expert_action, check = env.step(t)
                if check:
                    break
                obs.append(env.observation['joint'])
                cv2.imwrite(str(i)+'/img1/'+str(t)+'.jpg', env.observation['image1'])
                cv2.imwrite(str(i)+'/img2/'+str(t)+'.jpg', env.observation['image2'])
                acs.append(action)
                exp_acs.append(expert_action)
                env.current_state = env.observation['joint']
            data = {'inits': env.inits, 'observations': obs, 'actions': exp_acs, 'policy': acs}
            if len(obs) != 0:
                with open(str(i)+'/data.pkl', 'wb') as f:
                    pickle.dump(data, f)
        else:
            print("collision at initial or target pose")
    # print("Episode finished after {} timesteps.\tTotal reward: {}".format(t+1,total_reward))
    env.close()
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))
