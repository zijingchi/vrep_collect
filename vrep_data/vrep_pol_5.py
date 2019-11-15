from vrep_env import vrep_env, vrep
import os
#import cv2
import time
import pickle
from gym import spaces
from gym.utils import seeding
#from keras.models import load_model
from train.keras_potential import construct_potential
from train.training_imgless import simple_model
from processing.angle_dis import config_dis
from processing.fknodes import tipcoor
#from train.mdn import sample_from_output
#from processing.readDataFromFile import deanglize
import numpy as np

pi = np.pi


class UR5ValEnv(vrep_env.VrepEnv):
    metadata = {'render.modes': [], }

    def __init__(
            self,
            server_addr='127.0.0.1',
            server_port=19997,
            scene_path=None,
            random_seed=0,
            l2_thresh=0.05,
            modelfile=None,
            dof=5,
            askvrep=False
    ):

        vrep_env.VrepEnv.__init__(
            self,
            server_addr,
            server_port,
            scene_path,
        )

        self.obstacle_pos = 5 * np.ones(3)
        self.obstacle_ori = np.zeros(3)
        # All joints
        ur5_joints = ['UR5_joint1', 'UR5_joint2', 'UR5_joint3', 'UR5_joint4', 'UR5_joint5', 'UR5_joint6'][:5]

        # Getting object handles
        self.obstacle = self.get_object_handle('Obstacle')
        # Meta
        self.goal_viz = self.get_object_handle('Cuboid')
        self.tip = self.get_object_handle('tip')
        self.distance_handle = self.get_distance_handle('Distance')
        self.distance = -1

        self.dof = dof
        self.askvrep = askvrep
        # Actuators
        self.oh_joint = list(map(self.get_object_handle, ur5_joints))
        self.init_joint_pos = np.array([0, -pi / 6, -3 * pi / 4, 0, pi / 2, 0])
        self.target_joint_pos = np.array([0, - pi / 6, - pi / 3, 0, pi / 2, 0])
        self.l2_thresh = l2_thresh
        self.collision_handle = self.get_collision_handle('Collision1')
        # self.self_col_handle = self.get_collision_handle('SelfCollision')
        joint_space = np.ones(self.dof)
        self.action_space = spaces.Box(low=-0.1 * joint_space, high=0.1 * joint_space)
        # self._make_obs_space()
        #self.model = simple_model(5, 25)
        self.model = construct_potential()
        self.seed(random_seed)
        if modelfile:
            self.model.load_weights(modelfile)
        self._make_obs_space()

    def _make_obs_space(self):
        joint_lbound = np.array([-2 * pi / 3, -pi / 2, -pi, -pi / 2, 0])
        joint_hbound = np.array([2 * pi / 3, pi / 6, 0, pi / 2, pi])
        obstacle_pos_lbound = np.array([-5, -5, 0])
        obstalce_pos_hbound = np.array([5, 5, 2])
        self.observation_space = spaces.Box(low=np.concatenate([joint_lbound, joint_lbound, obstacle_pos_lbound]),
                                                high=np.concatenate([joint_hbound, joint_hbound, obstalce_pos_hbound]))

    def _make_observation(self):
        """Get observation from v-rep and stores in self.observation
        """
        joint_angles = [self.obj_get_joint_angle(joint) for joint in self.oh_joint]
        self.distance = self.read_distance(self.distance_handle)
        self.tip_pos = self.obj_get_position(self.tip)
        self.observation = np.concatenate([np.array(joint_angles).astype('float32'),
                                           self.target_joint_pos,
                                           self.obstacle_pos,
                                           tipcoor(joint_angles)[3:-3]
                                           #np.array([self.distance])
                                           ])

        return self.observation

    def _config(self):
        return self.observation[:5]

    def _reshape_observation(self):
        config = self._config()
        xyzs = tipcoor(config)[3:-3]
        matrix = np.zeros((5, 5))
        matrix[0, :] = config
        matrix[1, :] = self.target_joint_pos
        matrix[2:, 0] = self.obstacle_pos
        matrix[2:, 1:] = xyzs.reshape((-1, 3)).T
        return matrix

    def set_joints(self, angles):
        for j, a in zip(self.oh_joint, angles):
            self.obj_set_joint_position(j, a)

    def _make_action(self, a):
        """Send action to v-rep
        """
        cfg = self._config()
        newa = a + cfg
        self.set_joints(newa)
        if (self.observation_space.low[:5]<newa).all() and (self.observation_space.high[:5]>newa).all():
            return True
        else:
            return False

    def directly_towards(self, init_jp, n):
        #init_jp = self.init_joint_pos
        sub = (self.target_joint_pos - init_jp) / n
        self.linear_sub = []
        for i in range(n):
            next_state = init_jp + sub * (i + 1)
            self.linear_sub.append(next_state)
            self.set_joints(next_state)
            self.step_simulation()
            colcheck = self.read_collision(self.collision_handle)
            if colcheck == 1:
                #print('colliding during direct path')
                return 1
        #print('reaching by direct path')
        return 0

    def reset(self):
        if self.sim_running:
            self.stop_simulation()
        while self.sim_running:
            self.stop_simulation()
        init_w = [0.1, 0.1, 0.1, 0.2, 0.2]
        self.init_joint_pos = np.array([0, -pi / 6, -3 * pi / 4, 0, pi / 2])
        self.target_joint_pos = np.array([0, - pi / 3, - pi / 3, 0, pi / 2])
        init_joint_pos = self.init_joint_pos + np.multiply(init_w, np.random.randn(5))
        target_joint_pos = self.target_joint_pos  # + np.multiply(init_w, np.random.randn(5))
        self.start_simulation()
        while abs(init_joint_pos[2]) > 5 * pi / 6 or abs(target_joint_pos[2]) > 5 * pi / 6:
            init_joint_pos[2] = self.init_joint_pos[2] + init_w[2] * np.random.randn()
            target_joint_pos[2] = self.target_joint_pos[2] + init_w[2] * np.random.randn()
        self.set_joints(target_joint_pos)
        self.init_joint_pos = init_joint_pos
        self.target_joint_pos = target_joint_pos
        self.init_goal_dis = self._angle_dis(init_joint_pos, target_joint_pos, 5)
        self.reset_obstacle()

        tip_pos = self.obj_get_position(self.tip)
        tip_ori = self.obj_get_orientation(self.tip)
        self.obj_set_position(self.goal_viz, tip_pos)
        self.obj_set_orientation(self.goal_viz, tip_ori)

        while self._clear_obs_col():
            self.reset_obstacle()

        self.step_simulation()
        ob = self._make_observation()

        return ob

    def _check_collision(self, pos, handle):
        self.set_joints(pos)
        self.step_simulation()
        return self.read_collision(handle)

    def reset_obstacle(self):
        init_tip = tipcoor(self.init_joint_pos)[-3:]
        goal_tip = tipcoor(self.target_joint_pos)[-3:]
        alpha = 0.4 + 0.2*np.random.randn()
        #obs_pos = alpha * init_tip + (1 - alpha) * goal_tip
        #obs_pos += np.concatenate((0.15 * np.random.randn(2), np.array([0.15 * np.random.rand() + 0.24])))
        obs_pos = alpha * init_tip + (1 - alpha) * goal_tip
        obs_pos += np.concatenate((0.1 * np.random.randn(2), np.array([0.25 * np.random.rand() - 0.01])))
        obs_pos[0] += 0.10*np.random.rand()
        self.obstacle_pos = np.clip(obs_pos, self.observation_space.low[5*2:5*2+3],
                                    self.observation_space.high[5*2:5*2+3])

    def _clear_obs_col(self):
        self.step_simulation()
        self.obj_set_position(self.obstacle, self.obstacle_pos)
        self.obj_set_orientation(self.obstacle, self.obstacle_ori)
        col1 = self._check_collision(self.target_joint_pos, self.collision_handle)
        col2 = self._check_collision(self.init_joint_pos, self.collision_handle)
        return col1 or col2

    def _action_process(self, ac):
        action = ac/np.linalg.norm(ac)*self.l2_thresh
        return action

    def _model_input(self):
        x1 = np.expand_dims(self.observation[:5], 0)
        x2 = np.expand_dims(self.observation[5:10], 0)
        x3 = np.expand_dims(self.observation[10:13], 0)
        x4 = np.expand_dims(self.observation[13:], 0)
        return [x1, x2, x3, x4]

    def step(self, t):
        self._make_observation()
        #matrix = np.expand_dims(self._reshape_observation(), -1)
        #ac = self.model.predict(np.expand_dims(self.observation, 0))
        ac = self.model.predict(self._model_input())
        ac = self._action_process(ac[0])

        cfg = self._config()
        info = {}
        if self.askvrep:
            inFloats = cfg.tolist() + self.target_joint_pos.tolist()
            minConfigs = int(100 * np.linalg.norm(self.target_joint_pos - cfg))
            dir_suc = self.directly_towards(cfg, int(minConfigs/5))
            if dir_suc==0:
                dif = self.target_joint_pos-self.init_joint_pos
                info["exp_ac"] = dif/np.linalg.norm(dif)*self.l2_thresh
            else:
                n_path, path, res = self._calPathThroughVrep(self.cID, minConfigs, inFloats, bytearray())
                if (res == 0) & (n_path != 0):
                    np_path = np.array(path)
                    re_path = np_path.reshape((n_path, 5))
                    for p in re_path:
                        n = np.linalg.norm(p - cfg)
                        if n > self.l2_thresh:
                            expert_action = p - cfg
                            info["exp_ac"] = expert_action
                            break
                elif res == 2:
                    print('timeout')
                    time.sleep(6)
                else:
                    print('no')

            self.set_joints(cfg)
        invalid = not self._make_action(ac)
        self.step_simulation()
        self._make_observation()
        self.collision_check = self.read_collision(self.collision_handle) or abs(self.observation[2]) > 5 * pi / 6

        done = self._angle_dis(cfg, self.target_joint_pos,
                               5) < 1.5 * self.l2_thresh or self.collision_check or invalid

        reward = self.compute_reward(cfg, ac)

        if reward > 0.5 and done:
            info["status"] = 'reach'
        elif reward < 0.5 and done:
            info["status"] = 'collide'
        else:
            info["status"] = 'running'

        return self.observation, reward, done, info

    def compute_reward(self, state, action):
        config_dis = self._angle_dis(state, self.target_joint_pos, 5)
        #pre_config_dis = self._angle_dis(state-action, self.target_joint_pos, 5)
        approach = 2 if config_dis < 1.5*self.l2_thresh else 0
        collision = -1 if self.collision_check else 0
        valid = (self.observation_space.low[:5]<state).all() and (self.observation_space.high[:5]>state).all()
        invalid = -1 if not valid else 0
        return approach + collision + invalid

    def _angle_dis(self, a1, a2, dof):
        return np.linalg.norm(a1[:dof]-a2[:dof])

    def _calPathThroughVrep(self, clientID, minConfigsForPathPlanningPath, inFloats, emptyBuff):
        """send the signal to v-rep and retrieve the path tuple calculated by the v-rep script"""
        dof = self.dof  # we will try to find 10 different states corresponding to the goal pose and order them according to distance from initial state
        maxTrialsForConfigSearch = 300  # a parameter needed for finding appropriate goal states
        searchCount = 1  # how many times OMPL will run for a given task
        # minConfigsForPathPlanningPath = 50  # interpolation states for the OMPL path
        minConfigsForIkPath = 80  # interpolation states for the linear approach path
        collisionChecking = 1  # whether collision checking is on or off
        inInts = [collisionChecking, minConfigsForIkPath, minConfigsForPathPlanningPath,
                  dof, maxTrialsForConfigSearch, searchCount]
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

    def render(self, mode='human', close=False):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


def main(args):
    #workpath = os.path.expanduser('~/Downloads/ur5expert3')
    path0 = os.getcwd()
    path1 = path0[:path0.rfind('/')]
    model_path = os.path.join(path1, 'train/h5files/pol_cpt1.h5')
    ask_vrep = False
    env = UR5ValEnv(modelfile=model_path, askvrep=ask_vrep, random_seed=7, l2_thresh=0.1)
    save_path = './dagger'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        maxdir = 0
    else:
        dirlist = os.listdir(save_path)
        numlist = [int(s) for s in dirlist if os.path.isdir(os.path.join(save_path, s))]
        if len(numlist) == 0:
            maxdir = 0
        else:
            maxdir = max(numlist)
    os.chdir(save_path)
    success = 0
    dsuccess = 0
    n = 100
    for i in range(1, n+1):
        print('iter:', i)
        env.reset()
        if ask_vrep:
            exp_acs = []
            obs = []
        for t in range(100):
            ob, rew, done, info = env.step(t)
            if ask_vrep and 'exp_ac' in info:
                exp_acs.append(info['exp_ac'])
                obs.append(ob[:5])
            if done:
                if info['status']=='reach':
                    success += 1
                dres = env.directly_towards(env.init_joint_pos, 30)
                if ask_vrep:
                    data = {'inits':{'init_joint_pos': env.init_joint_pos, 'target_joint_pos': env.target_joint_pos,
                                     'obstacle_pos': env.obstacle_pos},
                            'observations': obs, 'actions': exp_acs
                            }
                    os.mkdir(str(maxdir+i))
                    with open(os.path.join(str(maxdir+i), 'data.pkl'), 'wb') as f:
                        pickle.dump(data, f)
                if dres==0:
                    dsuccess += 1
                break
    print('policy success rate: {}/{}'.format(success, n))
    print('direct success rate: {}/{}'.format(dsuccess, n))

    env.close()
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))
