from vrep_env import vrep_env, vrep
import time
import numpy as np

pi = np.pi

class FuncTests(vrep_env.VrepEnv):
    def __init__(self,
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

        ur5_joints = ['UR5_joint1', 'UR5_joint2', 'UR5_joint3', 'UR5_joint4', 'UR5_joint5']
        # All shapes
        # ur5_links_visible = ['UR5_link1_visible','UR5_link2_visible','UR5_link3_visible',
        #	'UR5_link4_visible','UR5_link5_visible','UR6_link1_visible','UR5_link7_visible']

        # Getting object handles
        self.obstable = self.get_object_handle('Obstacle')
        self.oh_joint = list(map(self.get_object_handle, ur5_joints))
        self.distance_handle = self.get_distance_handle('Distance')
        self.joint_pos = np.array([0, -pi / 12, -3 * pi / 4, 0, pi / 2])

    def step(self, action):
        self.set_joints(action)
        self.step_simulation()

    def set_joints(self, angles):
        for j, a in zip(self.oh_joint, angles):
            self.obj_set_joint_position(j, a)

    def render(self, mode='human'):
        pass


def main():
    env = FuncTests()
    env.start_simulation()
    ac = 0.1*np.ones(5)
    for i in range(10):
        env.joint_pos += ac
        env.step(env.joint_pos)
        dis = env.read_distance(env.distance_handle)
        print(dis)
        time.sleep(0.1)
    env.close()


if __name__ == '__main__':
    main()
