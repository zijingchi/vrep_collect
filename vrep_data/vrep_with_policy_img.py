import os
import cv2
import time
import pickle
from keras.models import load_model
from vrep_data.vrep_with_policy1 import UR5DaggerSample
import numpy as np

pi = np.pi


class ImgPolicyExe(UR5DaggerSample):

    def __init__(
            self,
            server_addr='127.0.0.1',
            server_port=19997,
            scene_path=None,
            modelfile=None,
            askvrep=True
    ):
        super(ImgPolicyExe, self).__init__(server_addr, server_port, scene_path, modelfile, askvrep)
        self.model = None
        if modelfile:
            from train.trainingTest1 import merge_model2
            self.model = merge_model2()
            self.model.load_weights(modelfile)

    def _model_input(self):
        x1 = self._adddim(self.observation['joint'][:3])
        x2 = self._adddim(self.target_joint_pos[:3])
        x3 = self._adddim(self.observation['image1'])
        x4 = self._adddim(self.observation['image2'])
        return [x1, x2, x3, x4]


def main(args):
    path0 = os.getcwd()
    hi = path0.find('home') + 5
    homepath = path0[:path0.find('/', hi)]
    workpath = homepath+'/vdp/5/'
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