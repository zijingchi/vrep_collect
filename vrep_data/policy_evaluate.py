from vrep_env import vrep_env, vrep
import os
import cv2
import time
import pickle
from vrep_data.collect_from_vrep1 import UR5WithCameraSample
import numpy as np

pi = np.pi


class PolicyEva(UR5WithCameraSample):

    def __init__(
            self,
            server_addr='127.0.0.1',
            server_port=19997,
            scene_path=None,
            datapath=None

    ):
        UR5WithCameraSample.__init__(
            self,
            server_addr,
            server_port,
            scene_path,
        )

        self.subdirs = os.listdir(datapath)
        np.random.shuffle(self.subdirs)

        print('PolicyEva: initialized')

    def reset_config(self, datapkl):
        if self.sim_running:
            self.stop_simulation()
        while self.sim_running:
            self.stop_simulation()

        with open(datapkl, 'rb') as f:
            data = pickle.load(f)
            obs_config = data['observations']
            actions = data['actions']
            inits = data['inits']

        self.start_simulation()
        self.init_joint_pos = obs_config[0]
        self.target_joint_pos = inits['target_joint_pos']
        self.set_joints(self.target_joint_pos)
        tip_pos = self.obj_get_position(self.tip)
        tip_ori = self.obj_get_orientation(self.tip)
        self.obj_set_position(self.goal_viz, tip_pos)
        self.obj_set_orientation(self.goal_viz, tip_ori)

        #self.set_joints(self.init_joint_pos)
        self.obj_set_position(self.obstable, inits['obstacle_pos'])
        self.obj_set_orientation(self.obstable, inits['obstacle_ori'])
        self.expert_actions = actions
        self.configs = obs_config
        self.step_simulation()

    def step(self, t):
        self.set_joints(self.configs[t])
        self._make_observation()
        self.step_simulation()
        time.sleep(1)

        self._make_action(self.expert_actions[t])

        self.step_simulation()
        time.sleep(0.5)

        self.set_joints(self.configs[t])
        self.step_simulation()
        time.sleep(1)


def main():
    datapath = '/home/czj/vrep_path_dataset/2_1/'
    os.chdir(datapath)
    env = PolicyEva(datapath=datapath)
    for s in env.subdirs:
        datapkl = os.path.join(s, 'data.pkl')
        if os.path.exists(datapkl):
            env.reset_config(datapkl)
            n = len(env.expert_actions)
            for i in range(n):
                env.step(i)


if __name__ == '__main__':
    main()
