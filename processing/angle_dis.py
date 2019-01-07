import numpy as np

metrics = [1.2, 2, 1.6, 0.4, 0.3]


def config_dis(config1, config2):
    y = np.multiply(np.array(metrics), config2 - config1)
    return np.linalg.norm(y)


def obs_pt(pos, ori):
    s = 0.05
    l = 0.3
    ps = s/2 * np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])
    ps = ps.transpose()
    P = np.empty((4, 8), dtype=np.double)
    for i in range(4):
        P[:2, i] = ps[:, i]
        P[:2, i+4] = ps[:, i]
    P[2, :4] = l/2 * np.ones((1, 4))
    P[2, 4:] = -l/2 * np.ones((1, 4))
    P[3, :] = np.ones((1, 8))
    R = euler2rotm(ori)
    pos.shape = (3, 1)
    T1 = np.concatenate((R, pos), axis=1)
    T = np.eye(4)
    T[:3, :4] = T1
    PafterT = T @ P
    PafterT = PafterT[:3, :].transpose()
    PafterT = np.reshape(PafterT, (1, -1))
    return PafterT


def euler2rotm(e):
    ax, ay, az = e[0], e[1], e[2]
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    Q = Rx @ Ry @ Rz
    return Q
