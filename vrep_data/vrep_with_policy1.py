from vrep_env import vrep_env, vrep
import os
import cv2
import time
import pickle
from keras.models import load_model
#from train.training_imgless2 import model_with_1dconv
from train.training_imgless import model_with_config_n_target2
from processing.angle_dis import obs_pt2, config_dis
from processing.fknodes import tipcoor
from vrep_data.collect_from_vrep1 import UR5WithCameraSample
import numpy as np

pi = np.pi


class UR5DaggerSample(UR5WithCameraSample):
    """
    regular policy model execution class
    """
    metadata = {'render.modes': [], }

    def __init__(
            self,
            server_addr='127.0.0.1',
            server_port=19997,
            scene_path=None,
            modelfile=None,
            askvrep=True
    ):

        UR5WithCameraSample.__init__(
            self,
            server_addr,
            server_port,
            scene_path,
        )

        self.model = None
        self.askvrep=askvrep
        if modelfile:
            #self.model = model_with_config_n_target2(3)
            #self.model.load_weights(modelfile)
            self.model = load_model(modelfile)

        #print('UR5DaggerSample: initialized')

    def _adddim(self, nparray):
        newa = np.empty(tuple([1]) + nparray.shape, float)
        newa[0] = nparray
        return newa

    def _model_input(self):
        x1 = self._adddim(self.observation['joint'][:3])
        x2 = self._adddim(self.target_joint_pos[:3])
        x3 = self._adddim(self.obstacle_pos)
        x4 = self._adddim(self.obstacle_ori)
        return [x1, x2, x3, x4]

    def _model_output_process(self, out, thresh):
        action = np.concatenate((out[0], np.zeros(2)))
        action = np.deg2rad(action)
        action = 1.3 * action + 1.1 * thresh * (self.target_joint_pos - self.observation['joint']) / config_dis(
            self.target_joint_pos, self.observation['joint'])
        return action

    def _model_output_process2(self, out, thresh):
        out = out[0]
        action = [np.sin(out[1])*np.cos(out[0]), np.sin(out[1])*np.sin(out[0]), np.cos(out[1])]
        action = np.concatenate((action, np.zeros(2)))
        action = 2*thresh * (self.target_joint_pos - self.observation['joint']) / config_dis(
            self.target_joint_pos, self.observation['joint']) + action/np.linalg.norm(action)*0.8*thresh
        return action

    def step(self, t):
        self._make_observation()    # make an observation
        action = np.zeros(5)
        # predict the action from the model
        config = self.observation['joint']
        emptyBuff = bytearray()
        thresh = 0.1
        if self.model:
            model_input = self._model_input()
            action = self.model.predict(model_input)
            action = self._model_output_process(action, thresh)

            self._make_action(action)   # make the action
            self.step_simulation()

        colcheck = self._checkInitCollision(self.cID, emptyBuff)
        amp_between = config_dis(self.target_joint_pos, config)
        expert_action = []
        check1 = amp_between < thresh
        check2 = colcheck == 1
        check = 0
        if check1:
            check = 1
            print('reaching')
        if check2:
            check = 2
            print('colliding')

        if self.askvrep:
        # ask the expert to give the right action if exists (here the expert is the ompl algorithm used in v-rep
            inFloats = config.tolist() + self.target_joint_pos.tolist()
            minConfigs = int(300 * np.linalg.norm(self.target_joint_pos - config))
            n_path, path, res = self._calPathThroughVrep(self.cID, minConfigs, inFloats, emptyBuff)

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
            if check3:
                check = 3
                print('expert action not found')
            #else:
            #    self._make_action(expert_action)

        self.step_simulation()

        return action, expert_action, check

    def cal_path_only(self):
        self._make_observation()  # make an observation
        config = self.observation['joint']
        inFloats = config.tolist() + self.target_joint_pos.tolist()
        minConfigs = int(100 * np.linalg.norm(self.target_joint_pos - config))
        emptyBuff = bytearray()
        n_path, path, res = self._calPathThroughVrep(self.cID, minConfigs, inFloats, emptyBuff)
        if (res == 0) & (n_path != 0):
            np_path = np.array(path)
            re_path = np_path.reshape((n_path, 5))
            return re_path
        elif res == 3:
            print('timeout')
            time.sleep(6)
            return []

    def reset(self):
        if self.sim_running:
            self.stop_simulation()
        while self.sim_running:
            self.stop_simulation()

        self.init_joint_pos = np.array([-pi/6, -pi / 12, -3 * pi / 4, 0, pi / 2])
        init_w = [0.4, 0.1, 0.1, 0.3, 0.3]
        for i in range(len(self.init_joint_pos) - 1):
            self.init_joint_pos[i] = self.init_joint_pos[i] + init_w[i] * np.random.randn()

        self.target_joint_pos = np.array([0.2*np.random.randn(), 0.1*np.random.randn()-pi/3,
        0.2*np.random.randn()-pi/3, 0.3*np.random.randn(), 0.2*np.random.randn()+pi/2])
        
        self.start_simulation()
        colcheck = self.set_obs_pos2()

        self.inits = {'target_joint_pos': self.target_joint_pos,
                      'obstacle_pos': self.obstacle_pos,
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
    workpath = homepath+'/vdp/4_6/'
    path1 = path0[:path0.rfind('/')]
    model_path = os.path.join(path1, 'train/h5files/3dof_model.h5')
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
    env = UR5DaggerSample(modelfile=model_path, askvrep=askvrep)
    i = maxdir + 1
    success = 0
    dsuccess = 0

    while i < maxdir + 11:
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
            check = -1
            for t in range(100):
                action, expert_action, check = env.step(t)
                if check:
                    break
                obs.append(env.observation['joint'])
                cv2.imwrite(str(i)+'/img1/'+str(t)+'.jpg', env.observation['image1'])
                cv2.imwrite(str(i)+'/img2/'+str(t)+'.jpg', env.observation['image2'])
                acs.append(action)
                if askvrep:
                    exp_acs.append(expert_action)
                env.current_state = env.observation['joint']
            if askvrep:
                data = {'inits': env.inits, 'observations': obs, 'actions': exp_acs, 'policy': acs}
            else:
                data = {'inits': env.inits, 'observations': obs, 'actions': acs}
            if len(obs) != 0:
                with open(str(i)+'/data.pkl', 'wb') as f:
                    pickle.dump(data, f)

            if check == 0:
                print('time end')
            elif check == 1:
                success = success + 1
            print('policy success rate:', success/(i-maxdir))
            dcheck = env.directly_towards(40)
            if dcheck == 0:
                dsuccess = dsuccess + 1
            print('linear path success rate:', dsuccess/(i-maxdir))
            i = i + 1
        else:
            print("collision at initial or target pose")
    # print("Episode finished after {} timesteps.\tTotal reward: {}".format(t+1,total_reward))
    env.close()
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))
