from vrep_env import vrep_env, vrep
import os
import pickle
import numpy as np
import time

pi = np.pi


class UR5WithCameraDagger(vrep_env.VrepEnv):
    metadata = {'render.modes': [], }

    def __init__(
            self,
            target_pos,
            obstacle_pos,
            obstacle_ori,
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
        self.target_joint_pos = target_pos
        self.obstacle_pos = obstacle_pos
        self.obstacle_ori = obstacle_ori
        # Actuators
        self.oh_joint = list(map(self.get_object_handle, ur5_joints))

        print('UR5VrepEnv: initialized')

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
                            'Dummy', vrep.sim_scripttype_childscript, 'findPath_goalIsState', 
                            inInts, inFloats, [], emptyBuff, vrep.simx_opmode_oneshot_wait)
        if (res == 0) and len(path) > 0:
            n_path = retInts[0]
        else:
            n_path = 0
        return n_path, path, res

    def set_obs_pos(self):
        self._set_joints(self.target_joint_pos)

        tip_pos = self.obj_get_position(self.tip)
        tip_ori = self.obj_get_orientation(self.tip)
        self.obj_set_position(self.goal_viz, tip_pos)
        self.obj_set_orientation(self.goal_viz, tip_ori)

        self.obj_set_position(self.obstable, self.obstacle_pos)
        self.obj_set_orientation(self.obstable, self.obstacle_ori)
        self.step_simulation()
        emptyBuff = bytearray()
        colcheck1 = self._checkInitCollision(self.cID, emptyBuff)

        if colcheck1 == 0:
            return 1
        else:
            return 0

    def _set_joints(self, angles):
        for j, a in zip(self.oh_joint, angles):
            self.obj_set_joint_position(j, a)

    def reset(self):
        while self.sim_running:
            self.stop_simulation()

        self.start_simulation()
        colcheck = self.set_obs_pos()
        self.step_simulation()
        return colcheck

    def _checkInitCollision(self, clientID, emptyBuff):
        res, retInts, path, retStrings, retBuffer = vrep.simxCallScriptFunction(clientID,
                            'Dummy', vrep.sim_scripttype_childscript, 'checkCollision',
                            [], [], [], emptyBuff, vrep.simx_opmode_oneshot_wait)
        if res == 0:
            return retInts[0]
        else:
            return -1

    def step(self, config):
        self._set_joints(config)
        self.step_simulation()
        inFloats = config.tolist() + self.target_joint_pos.tolist()
        minConfigs = int(50*np.linalg.norm(self.target_joint_pos - config)/1.35)
        emptyBuff = bytearray()

        n_path, path, res = self._calPathThroughVrep(self.cID, minConfigs, inFloats, emptyBuff)
        thresh = 0.06
        if (res == 0) & (n_path != 0):
            np_path = np.array(path)
            re_path = np_path.reshape((n_path, 6))
            for p in re_path:
                n = np.linalg.norm(p - config)
                if n > thresh:
                    return 0, p
        return res, []

    def render(self, mode='human', close=False):
        pass

    def calAngDis(self, angles, targetangles):
        return np.linalg.norm(angles - targetangles)


def main(args):
    workpath = '/home/ubuntu/vrep_path_dataset/9/'
    if not os.path.exists(workpath):
        raise RuntimeError('directory does not exist!')
    dirlist = os.listdir(workpath)
    os.chdir(workpath)

    for i in dirlist:
        print('dir:', i)
        datapkl = workpath + i + '/data.pkl'
        daggerpkl = workpath + i + '/dagger.pkl'
        if os.path.exists(daggerpkl):
            print('exist')
            continue
        with open(datapkl, 'rb') as f:
            data = pickle.load(f)
            configs = [obs['joint'] for obs in data['observations']]
            inits = data['inits']
            target = inits['target_joint_pos']
            obstacle_pos = inits['obstacle_pos']
            obstacle_ori = inits['obstacle_ori']
            env = UR5WithCameraDagger(target, obstacle_pos, obstacle_ori)
            colcheck = env.reset()
            if colcheck == 1:
                actions = {}
                for k in range(len(configs)):
                    res, nextc = env.step(configs[k])
                    if res == 3:
                        time.sleep(8)
                        break
                    if len(nextc) == 6:
                        actions[k] = nextc - configs[k]
                if actions != {}:
                    with open(daggerpkl, 'wb') as df:
                        pickle.dump(actions, df)
            env.close()

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))
