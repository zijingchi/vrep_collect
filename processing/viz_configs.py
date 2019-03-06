import numpy as np
import pickle
#import tensorflow as tf
#import tensorflow_probability as tfp
import matplotlib.pyplot as plt
#from train.mdn import *
import os
#tfd = tfp.distributions


def read_configs(pklfile, rad2deg=False):
    with open(pklfile, 'rb') as f:
        data = pickle.load(f)
        actions = np.array(data['actions'])
        configs = np.array(data['observations'])
        if rad2deg:
            actions = np.rad2deg(actions)
            configs = np.rad2deg(configs)
        return actions, configs


def statics_config(configs):
    variance = np.var(configs, axis=0)
    mean = np.mean(configs, axis=0)
    return mean, variance


def plot_configs(configs, i):
    a = configs[:, i]
    t = list(range(a.size))
    plt.plot(t, a, 'b')
    plt.show()


"""def mdn_test(config, num_mix):
    loss_func = get_mixture_loss_func(5, num_mix)
    x = tf.placeholder(shape=(11*num_mix,), dtype=tf.float32, name='params')
    a = tf.placeholder(shape=(5,), dtype=tf.float32, name='y_true')
    loss = loss_func(a, x)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    sess = tf.Session()
    tf.global_variables_initializer().run()
    for _ in range(50):
        sess.run(train_step, feed_dict={})"""


def read_obs(datapath):
    dirlist = os.listdir(datapath)
    obs_poses = []
    for sub in dirlist:
        pklname = os.path.join(datapath, sub+'/data.pkl')
        if os.path.exists(pklname):
            with open(pklname, 'rb') as f:
                data = pickle.load(f)
                obs_poses.append(data['inits']['obstacle_pos'])
    return np.array(obs_poses)


def main():
    path0 = os.getcwd()
    hi = path0.find('home') + 5
    homepath = path0[:path0.find('/', hi)]
    workpath = homepath + '/vdp/test/'
    obs_poses = read_obs(workpath)
    mean, var = np.mean(obs_poses, 0), np.var(obs_poses, 0)
    pklfile1 = os.path.join(workpath, '30/data.pkl')
    actions, configs = read_configs(pklfile1, True)
    #plot_configs(actions, 3)
    print(statics_config(actions))


if __name__ == '__main__':
    main()
