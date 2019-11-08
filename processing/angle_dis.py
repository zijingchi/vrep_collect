import numpy as np

metrics = [1.2, 1.2, 1.0, 0.2, 0.2]


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
    #PafterT = np.reshape(PafterT, (1, -1))
    return PafterT


def euler2rotm(e):
    ax, ay, az = e[0], e[1], e[2]
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    Q = Rx @ Ry @ Rz
    return Q


def obs_pt2(pos, ori):
    s = 0.05
    l = 0.51
    h = 0.2
    ps = 1/2 * np.array([[s, l], [s, -l], [-s, -l], [-s, l]])
    ps = ps.transpose()
    P = np.empty((4, 8), dtype=np.double)
    for i in range(4):
        P[:2, i] = ps[:, i]
        P[:2, i+4] = ps[:, i]
    P[2, :4] = h/2 * np.ones((1, 4))
    P[2, 4:] = -h/2 * np.ones((1, 4))
    # P[2, 4:] = np.zeros((1, 4))
    P[3, :] = np.ones((1, 8))
    R = euler2rotm(ori)
    pos.shape = (3, 1)
    T1 = np.concatenate((R, pos), axis=1)
    T = np.eye(4)
    T[:3, :4] = T1
    PafterT = T @ P
    PafterT = PafterT[:3, :].transpose()
    #PafterT = np.reshape(PafterT, (1, -1))
    return PafterT


def obs_pt_big(pos, ori):
    s = 0.05
    l = 0.51
    h = 0.2
    deta = 1e-2
    n1, n2, n3 = int(s/deta)+1, int(l/deta)+1, int(h/deta)+1
    xs = np.linspace(-s/2, s/2, n1)
    ys = np.linspace(-l/2, l/2, n2)
    zs = np.linspace(-h/2, h/2, n3)
    P = np.ones((4, n1*n2*n3))
    i = 0
    for i1, x in enumerate(xs):
        for i2, y in enumerate(ys):
            for i3, z in enumerate(zs):
                P[:3, i] = np.array([xs[i1], ys[i2], zs[i3]])
                i = i + 1
    R = euler2rotm(ori)
    pos.shape = (3, 1)
    T = np.concatenate((R, pos), axis=1)
    PafterT = T @ P
    return PafterT.transpose()


def cal_avo_dir(action, tar, cur, thresh, dof):
    #avo = action - config_dis(action, np.zeros(5))*(tar - cur)/config_dis(tar, cur)
    avo = action - thresh * (tar - cur) / config_dis(tar, cur)
    #print(config_dis(action, np.zeros(5)))
    #avo = action - thresh * (tar - cur) / np.linalg.norm(tar[:dof]-cur[:dof])
    return avo[:dof]
