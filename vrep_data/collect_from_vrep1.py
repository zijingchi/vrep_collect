from vrep_env import vrep_env, vrep
import os
import cv2
import time
import pickle

# import gym
# from gym import spaces
# from gym.utils import seeding
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
        self.init_joint_pos = [0, -pi / 12, -2 * pi / 3, 0, pi / 2, 0]
        for i in range(len(self.init_joint_pos)):
            self.init_joint_pos[i] = self.init_joint_pos[i] + 0.05 * np.random.randn()
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
        searchCount = 8  # how many times OMPL will run for a given task
        # minConfigsForPathPlanningPath = 50  # interpolation states for the OMPL path
        minConfigsForIkPath = 100  # interpolation states for the linear approach path
        collisionChecking = 1  # whether collision checking is on or off
        inInts = [collisionChecking, minConfigsForIkPath, minConfigsForPathPlanningPath,
                  maxConfigsForDesiredPose, maxTrialsForConfigSearch, searchCount]
        res, retInts, path, retStrings, retBuffer = vrep.simxCallScriptFunction(clientID,
                            'Dummy', vrep.sim_scripttype_childscript, 'findPath_goalIsState',
                            inInts, inFloats, [], emptyBuff, vrep.simx_opmode_oneshot_wait)
        if (res == 0) and len(path) > 0:
            n_path = retInts[0]
        else:
            n_path = 0
        return n_path, path, res

    def _checkInitCollision(self, clientID, emptyBuff):
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
        self.set_joints(self.target_joint_pos)

        tip_pos = self.obj_get_position(self.tip)
        tip_ori = self.obj_get_orientation(self.tip)
        self.obj_set_position(self.goal_viz, tip_pos)
        self.obj_set_orientation(self.goal_viz, tip_ori)
        # obstacle_pos = tip_pos + 0.2*(np.random.rand(3) + np.array([-3.3, 1.1, -0.3]))
        self.obstacle_pos = 0.75*np.array(tip_pos) + np.array([0.1*np.random.randn(),
                                                              0.1+0.1*(0.5-np.random.rand()),
                                                              0.15+0.1*np.random.rand()])
        #self.obstacle_pos[2] = 0.35 + 0.2*np.random.rand()
        self.obstacle_pos = np.array(self.obstacle_pos)
        self.obj_set_position(self.obstable, self.obstacle_pos)
        self.obstacle_ori = 0.2 * np.random.rand(3)
        self.obj_set_orientation(self.obstable, self.obstacle_ori)
        emptyBuff = bytearray()
        colcheck1 = self._checkInitCollision(self.cID, emptyBuff)
        self.set_joints(self.init_joint_pos)
        colcheck2 = self._checkInitCollision(self.cID, emptyBuff)
        if ((colcheck1 == 0) & (colcheck2 == 0)):
            return 1
        else:
            return 0

    def step(self, t):
        self._make_observation()
        next_state = np.append(self.path[t+1], np.array([0]))
        action = next_state - self.observation['joint']
        print(np.linalg.norm(action))
        #self.set_joints(next_state)
        self._make_action(action)
        self.step_simulation()

        return self.observation, action[:-1]

    def reset(self):
        if self.sim_running:
            self.stop_simulation()
        while self.sim_running:
            self.stop_simulation()
        self.target_joint_pos = [0.2 * np.random.randn(), 0.1 * np.random.randn() - pi / 4,
                                0.2 * np.random.randn() - pi / 4, 0.3 * np.random.randn(),
                                0.2 * np.random.randn() + pi / 2, 0]
        self.start_simulation()
        found = False

        colcheck = self.set_obs_pos()
        self.inits = {'target_joint_pos': np.array(self.target_joint_pos),
                      'init_joint_pos': np.array(self.init_joint_pos),
                      'obstacle_pos': np.array(self.obstacle_pos),
                      'obstacle_ori': self.obstacle_ori}
        if colcheck == 1:
            self.set_joints(self.init_joint_pos)
            self.step_simulation()
            self.current_state = np.array(self.init_joint_pos).astype('float32')
            clientID = self.cID
            inFloats = self.init_joint_pos + self.target_joint_pos
            emptyBuff = bytearray()
            n_path, path, res = self._calPathThroughVrep(clientID, 400, inFloats, emptyBuff)
            if (res == 0) & (n_path != 0):
                np_path = np.array(path)
                re_path = np_path.reshape((n_path, 5))
                # re_path = re_path[:, 0:5]
                thresh = 0.08
                c0 = np.array(self.init_joint_pos[:-1])
                final_path = [c0]
                for c in re_path:
                    if self.calAngDis(c, c0) > thresh:
                        final_path.append(c)
                        c0 = c
                if c0.any() != np.array(self.target_joint_pos[:-1]).any():
                    final_path.append(np.array(self.target_joint_pos[:-1]))
                self.n_path = len(final_path)
                self.path = final_path
                # print('obstacle_pos:', self.obstacle_pos)
                found = True
            if res == 3:
                time.sleep(5)

        return found

    def render(self, mode='human', close=False):
        pass

    def calAngDis(self, angles, targetangles):
        return np.linalg.norm(angles - targetangles)


def main(args):
    path0 = os.getcwd()
    hi = path0.find('home') + 5
    homepath = path0[:path0.find('/', hi)]
    workpath = homepath + '/vrep_path_dataset/16/'
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
    for i in range(maxdir + 1, maxdir + 50):
        print('iter:', i)
        found = env.reset()
        if found:
            os.mkdir(str(i))
            os.mkdir(str(i) + "/img1")
            os.mkdir(str(i) + "/img2")
            obs = []
            acs = []
            for t in range(env.n_path-1):
                observation, action = env.step(t)
                obs.append(observation['joint'][:-1])
                acs.append(action)
                img1_path = str(i) + "/img1/" + str(t) + ".jpg"
                img2_path = str(i) + "/img2/" + str(t) + ".jpg"
                cv2.imwrite(img1_path, observation['image1'])
                cv2.imwrite(img2_path, observation['image2'])
            data = {'inits': env.inits, 'observations': obs, 'actions': acs}
            with open(workpath + str(i) + '/data.pkl', 'wb') as f:
                pickle.dump(data, f)
        else:
            print("path not found")
    # print("Episode finished after {} timesteps.\tTotal reward: {}".format(t+1,total_reward))
    env.close()
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))
