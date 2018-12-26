from vrep_env import vrep_env, vrep
import os
import cv2
import time
import pickle
from keras.models import load_model
from train.mdn import *
import numpy as np

pi = np.pi


class UR5WithCameraSample(vrep_env.VrepEnv):
    metadata = {'render.modes': [], }

    def __init__(
            self,
            server_addr='127.0.0.1',
            server_port=19997,
            scene_path=None,
            modelfile=None
    ):

        vrep_env.VrepEnv.__init__(
            self,
            server_addr,
            server_port,
            scene_path,
        )

        # Settings
        self.random_start = False

        # All joints
        ur5_joints = ['UR5_joint1', 'UR5_joint2', 'UR5_joint3', 'UR5_joint4', 'UR5_joint5', 'UR5_joint6']
        # All shapes
        # ur5_links_visible = ['UR5_link1_visible','UR5_link2_visible','UR5_link3_visible',
        #	'UR5_link4_visible','UR5_link5_visible','UR6_link1_visible','UR5_link7_visible']

        # Getting object handles
        self.obstable = self.get_object_handle('Obstacle')
        # Meta
        self.camera1 = self.get_object_handle('camera1')
        self.zfar1 = self.get_obj_float_parameter(self.camera1,
                                                  vrep.sim_visionfloatparam_far_clipping)
        self.znear1 = self.get_obj_float_parameter(self.camera1,
                                                   vrep.sim_visionfloatparam_near_clipping)
        self.camera2 = self.get_object_handle('camera2')
        self.goal_viz = self.get_object_handle('Cuboid')
        self.tip = self.get_object_handle('tip')
        self.model = load_model(modelfile)
        h = 256
        w = 256
        c = 3
        self.img_size = [h, w, c]
        # self.target_joint_pos = [-pi/4, -pi/4, -pi/3, pi/8, pi/4, 0]

        self.init_joint_pos = np.array([0, -pi / 12, -2 * pi / 3, 0, pi / 2, 0]) + 0.01 * np.random.randn(6)
        # Actuators
        self.oh_joint = list(map(self.get_object_handle, ur5_joints))

        print('UR5VrepEnv: initialized')

    def _make_observation(self):
        """Get observation from v-rep and stores in self.observation
        """
        img1 = self.obj_get_vision_image(self.camera1)
        img2 = self.obj_get_vision_image(self.camera2)
        img1 = np.flip(img1, 2)
        img2 = np.flip(img2, 2)
        joint_angles = [self.obj_get_joint_angle(joint) for joint in self.oh_joint]
        depM1 = self._cal_depth(self.camera1, self.zfar1, self.znear1)

        self.observation = {'joint': np.array(joint_angles).astype('float32'),
                            'image1': img1, 'image2': img2}
        # self.observation = np.array(joint_angles).astype('float32')

    def _calPathThroughVrep(self, clientID, minConfigsForPathPlanningPath, inFloats, emptyBuff):
        """send the signal to v-rep and retrieve the path tuple calculated by the v-rep script"""
        maxConfigsForDesiredPose = 10  # we will try to find 10 different states corresponding to the goal pose and order them according to distance from initial state
        maxTrialsForConfigSearch = 300  # a parameter needed for finding appropriate goal states
        searchCount = 4  # how many times OMPL will run for a given task
        # minConfigsForPathPlanningPath = 50  # interpolation states for the OMPL path
        minConfigsForIkPath = 100  # interpolation states for the linear approach path
        collisionChecking = 1  # whether collision checking is on or off
        inInts = [collisionChecking, minConfigsForIkPath, minConfigsForPathPlanningPath,
                  maxConfigsForDesiredPose, maxTrialsForConfigSearch, searchCount]
        res, retInts, path, retStrings, retBuffer = vrep.simxCallScriptFunction(clientID,
                                                                                'Dummy',
                                                                                vrep.sim_scripttype_childscript,
                                                                                'findPath_goalIsState',
                                                                                inInts, inFloats, [], emptyBuff,
                                                                                vrep.simx_opmode_oneshot_wait)
        if (res == 0) and len(path) > 0:
            n_path = retInts[0]
        else:
            n_path = 0
        return n_path, path, res

    def _checkInitCollision(self, clientID, emptyBuff):
        res, retInts, path, retStrings, retBuffer = vrep.simxCallScriptFunction(clientID,
                                                                                'Dummy',
                                                                                vrep.sim_scripttype_childscript,
                                                                                'checkCollision',
                                                                                [], [], [], emptyBuff,
                                                                                vrep.simx_opmode_oneshot_wait)
        if res == 0:
            return retInts[0]
        else:
            return -1

    def _make_action(self, a):
        """Send action to v-rep
        """
        newa = a + self.observation['joint']
        self._set_joints(newa)

    def _cal_depth(self, chandle, zfar, znear):
        depmat = self.obj_get_depth_matrix(chandle)
        depmat = znear * np.ones(np.shape(depmat)) + (zfar - znear) * depmat
        return depmat

    def _set_joints(self, angles):
        for j, a in zip(self.oh_joint, angles):
            self.obj_set_joint_position(j, a)

    def set_obs_pos(self):
        self._set_joints(self.target_joint_pos)

        tip_pos = self.obj_get_position(self.tip)
        tip_ori = self.obj_get_orientation(self.tip)
        self.obj_set_position(self.goal_viz, tip_pos)
        self.obj_set_orientation(self.goal_viz, tip_ori)
        self.obstacle_pos = 0.8 * np.array(tip_pos) + np.array([0.1 * (0.5 - np.random.rand()),
                                                                0.1 - 0.15 * np.random.rand(),
                                                                0.3 + 0.2 * (0.5 - np.random.rand())])
        # self.obstacle_pos = [0.15*np.random.randn()-0.1, 0.2*np.random.randn()-0.45, 0.1*np.random.randn()+0.42]
        # self.obstacle_pos = np.array(self.obstacle_pos)
        self.obj_set_position(self.obstable, self.obstacle_pos)
        self.obstacle_ori = 0.3 * np.random.rand(3)
        self.obj_set_orientation(self.obstable, self.obstacle_ori)
        emptyBuff = bytearray()
        colcheck1 = self._checkInitCollision(self.cID, emptyBuff)
        self._set_joints(self.init_joint_pos)
        colcheck2 = self._checkInitCollision(self.cID, emptyBuff)
        if ((colcheck1 == 0) & (colcheck2 == 0)):
            return 1
        else:
            return 0

    def _transpose(self, nparray):
        m = np.mat(nparray).T
        m.shape = (1, nparray.size)
        return m

    def step(self, t):
        self._make_observation()  # make an observation
        # predict the action from the model
        config = self.observation['joint']
        x1 = self._transpose(np.rad2deg(config))
        x2 = self._transpose(np.rad2deg(self.target_joint_pos))
        x3 = self._transpose(np.concatenate((self.obstacle_pos, self.obstacle_ori)))
        param = self.model.predict([x1, x2, x3])
        action = sample_from_output(param, 6, 35)
        # action = action[0]
        action = np.deg2rad(action)
        self._make_action(action)  # make the action
        # ask the expert to give the right action if exists (here the expert is the ompl algorithm used in v-rep
        inFloats = config.tolist() + self.target_joint_pos.tolist()
        minConfigs = int(60 * np.linalg.norm(self.target_joint_pos - config) / 1.35)
        emptyBuff = bytearray()
        n_path, path, res = self._calPathThroughVrep(self.cID, minConfigs, inFloats, emptyBuff)
        thresh = 0.06
        expert_action = []
        if (res == 0) & (n_path != 0):
            np_path = np.array(path)
            re_path = np_path.reshape((n_path, 6))
            for p in re_path:
                n = np.linalg.norm(p - config)
                if n > thresh:
                    expert_action = p - config
                    break
        if res == 3:
            time.sleep(3)
        colcheck = self._checkInitCollision(self.cID, emptyBuff)
        amp_between = np.linalg.norm(self.target_joint_pos - config)
        check = (amp_between < 0.2) or (colcheck == 1) or (expert_action == [])
        self.step_simulation()

        return action, expert_action, check

    def reset(self):
        if self.sim_running:
            self.stop_simulation()
        while self.sim_running:
            self.stop_simulation()

        self.target_joint_pos = np.array([0.2 * np.random.randn(), 0.1 * np.random.randn() - pi / 3,
                                          0.2 * np.random.randn() - pi / 3, 0.3 * np.random.randn(),
                                          0.2 * np.random.randn() + pi / 2, 0])

        self.start_simulation()
        colcheck = self.set_obs_pos()
        self.inits = {'target_joint_pos': self.target_joint_pos,
                      'obstacle_pos': np.array(self.obstacle_pos),
                      'obstacle_ori': self.obstacle_ori}
        self.step_simulation()
        if colcheck == 1:
            self.current_state = self.init_joint_pos

        return colcheck

    def render(self, mode='human', close=False):
        pass

    def calAngDis(self, angles, targetangles):
        return np.linalg.norm(angles - targetangles)


def main(args):
    workpath = '/home/czj/vrep_path_dataset/15/'
    model_path = '/home/czj/vrep_path_dataset/model8_3.h5'
    if not os.path.exists(workpath):
        os.mkdir(workpath)
    dirlist = os.listdir(workpath)
    numlist = [int(s) for s in dirlist]
    if len(numlist) == 0:
        maxdir = -1
    else:
        maxdir = max(numlist)
    os.chdir(workpath)
    env = UR5WithCameraSample(modelfile=model_path)
    for i in range(maxdir + 1, maxdir + 20):
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
            for t in range(60):
                action, expert_action, check = env.step(t)
                if check:
                    break
                obs.append(env.observation['joint'])
                cv2.imwrite(str(i) + '/img1/' + str(t) + '.jpg', env.observation['image1'])
                cv2.imwrite(str(i) + '/img2/' + str(t) + '.jpg', env.observation['image2'])
                acs.append(action)
                exp_acs.append(expert_action)
                env.current_state = env.observation
            data = {'inits': env.inits, 'observations': obs, 'actions': exp_acs, 'policy': acs}
            if len(obs) != 0:
                with open(str(i) + '/data.pkl', 'wb') as f:
                    pickle.dump(data, f)
        else:
            print("collision at target pose")
    # print("Episode finished after {} timesteps.\tTotal reward: {}".format(t+1,total_reward))
    env.close()
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))
