from vrep_env import vrep_env, vrep
import os
import cv2
import time
import pickle
from processing.angle_dis import config_dis
from processing.fknodes import tipcoor
import numpy as np

pi = np.pi


class UR5WithCameraSample(vrep_env.VrepEnv):
    metadata = {'render.modes': [], }

    def __init__(
            self,
            server_addr='127.0.0.1',
            server_port=19997,
            scene_path=None,
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
        ur5_joints = ['UR5_joint1', 'UR5_joint2', 'UR5_joint3', 'UR5_joint4', 'UR5_joint5']
        # All shapes
        # ur5_links_visible = ['UR5_link1_visible','UR5_link2_visible','UR5_link3_visible',
        #	'UR5_link4_visible','UR5_link5_visible','UR6_link1_visible','UR5_link7_visible']

        # Getting object handles
        self.obstable = self.get_object_handle('Table')
        # Meta
        self.camera1 = self.get_object_handle('camera1')
        self.zfar1 = self.get_obj_float_parameter(self.camera1,
                                                  vrep.sim_visionfloatparam_far_clipping)
        self.znear1 = self.get_obj_float_parameter(self.camera1,
                                                   vrep.sim_visionfloatparam_near_clipping)
        self.camera2 = self.get_object_handle('camera2')
        self.goal_viz = self.get_object_handle('Cuboid')
        self.tip = self.get_object_handle('tip')
        self.target = self.get_object_handle('target')
        self.init_joint_pos = [-pi/4, -pi / 12, -12 * pi / 12, 0, 0]

        h = 256
        w = 256
        c = 3
        self.img_size = [h, w, c]
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

        self.observation = {'joint': np.array(joint_angles).astype('float32'),
                            'image1': img1, 'image2': img2}

    def _calPathThroughVrep(self, clientID, minConfigsForPathPlanningPath, inFloats, emptyBuff):
        """send the signal to v-rep and retrieve the path tuple calculated by the v-rep script"""
        maxConfigsForDesiredPose = 10  # we will try to find 10 different states corresponding to the goal pose and order them according to distance from initial state
        maxTrialsForConfigSearch = 300  # a parameter needed for finding appropriate goal states
        searchCount = 3  # how many times OMPL will run for a given task
        # minConfigsForPathPlanningPath = 50  # interpolation states for the OMPL path
        minConfigsForIkPath = 100  # interpolation states for the linear approach path
        collisionChecking = 1  # whether collision checking is on or off
        inInts = [collisionChecking, minConfigsForIkPath, minConfigsForPathPlanningPath,
                  maxConfigsForDesiredPose, maxTrialsForConfigSearch, searchCount]
        res, retInts, path, retStrings, retBuffer = vrep.simxCallScriptFunction(clientID,
                            'Dummy', vrep.sim_scripttype_childscript, 'findPath_goalIsState',
                            inInts, inFloats, [], emptyBuff, vrep.simx_opmode_oneshot_wait)
        """retInts, path, retStrings, retBuffer = self.call_childscript_function('Dummy', 'findPath_goalIsState', 
                                                                              [inInts, inFloats, [], emptyBuff])"""
        #length = 0
        if (res == 0) and len(path) > 0:
            n_path = retInts[0]
            final = np.array(path[-5:])
            tar = np.array(inFloats[-5:])
            if np.linalg.norm(final - tar) > 0.01:
                n_path = 0
                path = []
            #length = retInts[2]
        else:
            n_path = 0
        return n_path, path, res

    def _checkInitCollision(self, clientID, emptyBuff):
        """returns 1 if collision occurred, 0 otherwise"""
        res, retInts, path, retStrings, retBuffer = vrep.simxCallScriptFunction(clientID,
                            'Dummy', vrep.sim_scripttype_childscript, 'checkCollision',
                            [], [], [], emptyBuff, vrep.simx_opmode_oneshot_wait)
        if res == 0:
            return retInts[0]
        else:
            return -1

    def _make_action(self, a):
        """Send action to v-rep
        """
        newa = a + self.observation['joint']
        self.set_joints(newa)

    def set_joints(self, angles):
        for j, a in zip(self.oh_joint, angles):
            self.obj_set_joint_position(j, a)

    def set_obs_pos(self):
        pass

    def step(self, t):
        self._make_observation()  # make an observation

        next_state = self.path[t + 1]
        action = next_state - self.observation['joint']
        self._make_action(action)
        self.step_simulation()

        return self.observation, action

    def gen_pos(self):
        x = -0.1 + 0.6*(0.5-np.random.rand())
        y = -0.7 + 0.6*(0.5-np.random.rand())
        z = 0.505
        return np.array([x, y, z])

    def gen_ori(self):
        ori = np.array([pi, 0, pi/2])
        mtc = [0.08, 0.08, 0.1]
        for i in range(3):
            ori[i] = ori[i] + mtc[i]
        return ori

    def chop_angle(self, config):
        for i in range(len(config)):
            theta = config[i]
            if -pi < (theta + 2*pi) < pi:
                config[i] = theta + 2*pi
            elif -pi < (theta - 2*pi) < pi:
                config[i] = theta - 2*pi
        return config

    def reset(self):
        if self.sim_running:
            self.stop_simulation()
        while self.sim_running:
            self.stop_simulation()

        init_w = [0.4, 0.4, 0.4, 0.4, 0.4]

        for i in range(len(self.init_joint_pos)):
            self.init_joint_pos[i] = self.init_joint_pos[i] + init_w[i] * np.random.randn()
        self.target_pos = self.gen_pos()
        self.target_ori = self.gen_ori()
        emptybuff = bytearray()
        self.start_simulation()
        fres = -2
        retInts, goalState, retStrings, retBuffer = self.call_childscript_function('Dummy', 'calGoalState', [[],
                                                    np.concatenate([self.target_pos, self.target_ori]).tolist(),
                                                    [], emptybuff])
        if retInts[0] == 1 and len(goalState) > 0:
            goalState = self.chop_angle(goalState)
            print(np.rad2deg(self.init_joint_pos))
            print(np.rad2deg(goalState[:5]))
            self.target_joint = goalState
            self.set_joints(goalState)
            self.obj_set_position(self.goal_viz, self.target_pos)
            self.obj_set_orientation(self.goal_viz, self.target_ori)
            colcheck1 = self._checkInitCollision(self.cID, emptybuff)
            self.set_joints(self.init_joint_pos)
            colcheck2 = self._checkInitCollision(self.cID, emptybuff)

            fres = -1 # -1 indicates collision in initial or target state
            if colcheck1 == 0 and colcheck2 == 0:
                self.step_simulation()
                inFloats = self.init_joint_pos + self.target_joint[:-1]
                n_path, path, res = self._calPathThroughVrep(self.cID, 400, inFloats, emptybuff)
                if (res == 0) and (n_path != 0):
                    np_path = np.array(path)
                    re_path = np_path.reshape((n_path, 5))
                    # re_path = re_path[:, 0:5]
                    thresh = 0.1
                    c0 = np.array(self.init_joint_pos)
                    final_path = [c0]
                    for c in re_path:
                        if config_dis(c, c0) > thresh:
                            final_path.append(c)
                            c0 = c
                    self.n_path = len(final_path)
                    self.path = final_path
                    fres = 0
                elif res == 3:
                    fres = 1
                    print('timeout')
                    time.sleep(5)
                else:
                    fres = 2

            self.inits = {'target_joint_pos': np.array(self.target_joint),
                          'target_pose': self.target_pos,
                          'target_ori': self.target_ori}
            self.step_simulation()
        else:
            print('calGoalState fail')
        return fres


def main(args):
    path0 = os.getcwd()
    hi = path0.find('home') + 5
    homepath = path0[:path0.find('/', hi)]
    workpath = homepath + '/vrep_path_dataset/7/'
    if not os.path.exists(workpath):
        os.mkdir(workpath)
    dirlist = os.listdir(workpath)
    numlist = [int(s) for s in dirlist]
    if len(numlist) == 0:
        maxdir = -1
    else:
        maxdir = max(numlist)
    os.chdir(workpath)
    env = UR5WithCameraSample()
    i = maxdir + 1
    while i < maxdir + 200:
        print('iter:', i)
        fres = env.reset()
        if fres == 0:
            print('path found')
            os.mkdir(str(i))
            os.mkdir(str(i) + "/img1")
            os.mkdir(str(i) + "/img2")
            obs = []
            acs = []
            for t in range(env.n_path-1):
                observation, action = env.step(t)
                obs.append(observation['joint'])
                acs.append(action)
                img1_path = str(i) + "/img1/" + str(t) + ".jpg"
                img2_path = str(i) + "/img2/" + str(t) + ".jpg"
                cv2.imwrite(img1_path, observation['image1'])
                cv2.imwrite(img2_path, observation['image2'])
            data = {'inits': env.inits, 'observations': obs, 'actions': acs}
            with open(workpath + str(i) + '/data.pkl', 'wb') as f:
                pickle.dump(data, f)
            i = i + 1
        elif fres == -1:
            print("collision at init or target")

    # print("Episode finished after {} timesteps.\tTotal reward: {}".format(t+1,total_reward))
    env.close()
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))
