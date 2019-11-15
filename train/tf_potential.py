import tensorflow as tf
import numpy as np
from processing.data_loader import ExpertDataset
import os
import pickle

datapath = '/home/czj/Downloads/ur5expert'

def ortho_init(scale=1.0):
    def _ortho_init(shape):
        shape = tuple(shape)
        if len(shape)==2:
            flat_shape = shape
        elif len(shape)==4:
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape==flat_shape else v
        q = q.reshape(shape)
        return (scale*q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


def fc(x, scope, nh, init_scale=1.0, init_bias=0.0, l2_regu_coef=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(l2_regu_coef)(w))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b


def po_construct(config, goal, pobs, fkps, acs, l2_reg_coef, lr, adam_epsilon=1e-6):
    is_training = tf.placeholder_with_default(False, (), 'is_training')
    Uattr = tf.square(tf.norm(config-goal, axis=1))
    x = tf.layers.dense(config, 128, kernel_initializer=ortho_init(1.), name='config-dense1',
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_coef))
    x = tf.layers.batch_normalization(x, training=is_training, name='config-bn1')
    x = tf.nn.leaky_relu(x, name='config-relu1')
    fk = tf.layers.dense(fkps, 128, kernel_initializer=ortho_init(1.), name='fk-dense1',
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_coef))
    fk = tf.layers.batch_normalization(fk, training=is_training, name='fk-bn1')
    fk = tf.nn.leaky_relu(fk, name='fk-relu1')

    obs = tf.layers.dense(pobs, 256, kernel_initializer=ortho_init(1.), name='obs-dense1',
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_coef))
    obs = tf.layers.batch_normalization(obs, training=is_training, name='obs-bn1')
    obs = tf.nn.leaky_relu(obs, name='obs-relu1')
    concat1 = tf.concat([x, fk, obs], axis=-1, name='concat1')
    y = tf.layers.dense(concat1, 256, kernel_initializer=ortho_init(1.), name='y-dense1',
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_coef))
    y = tf.layers.batch_normalization(y, training=is_training, name='y-bn1')
    y = tf.nn.leaky_relu(y, name='y-relu1')
    y = tf.layers.dense(y, 128, kernel_initializer=ortho_init(1.), name='y-dense2',
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_coef))
    y = tf.layers.batch_normalization(y, training=is_training, name='y-bn2')
    y = tf.nn.leaky_relu(y, name='y-relu2')
    Urep = tf.layers.dense(y, 1, name='urep',
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_coef))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        Urep = tf.identity(Urep)
        U = Urep + Uattr
        grad = -tf.gradients(U, config)
        loss = tf.reduce_sum(tf.square(tf.nn.l2_normalize(grad)-tf.nn.l2_normalize(acs)))
        losses = loss + tf.losses.get_regularization_loss()
        AdamOp = tf.train.AdamOptimizer(learning_rate=lr, epsilon=adam_epsilon).minimize(losses)

    return [Urep, grad, loss, AdamOp]


def train(lr, l2_reg_coef, epochs, batch_size):
    loader = ExpertDataset(datapath)
    '''with open(os.path.join(datapath, 'list1.pkl'), 'rb') as f:
        lists = pickle.load(f)
        train_list = lists['train']
        vali_list = lists['test']'''
    train_iters = loader.n_train//batch_size

    cfg = tf.placeholder(dtype=tf.float32, shape=(None, 5), name='config')
    g = tf.placeholder(dtype=tf.float32, shape=(None, 5), name='goal')
    pobs = tf.placeholder(dtype=tf.float32, shape=(None, 3), name='obstacle')
    fkps = tf.placeholder(dtype=tf.float32, shape=(None, 12), name='fkpoints')
    acs = tf.placeholder(dtype=tf.float32, shape=(None, 5), name='actions')
    [Urep, grad, loss, AdamOp] = po_construct(cfg, g, pobs, fkps, acs, l2_reg_coef, lr)
    with tf.Session() as sess:
        for i in range(epochs):
            for j in range(train_iters):
                ob, ac = loader.get_next_batch(batch_size, True)
                sess.run(AdamOp, feed_dict={cfg: ob[:, :5], g: ob[:, 5:10],
                    pobs: ob[:, 10:13], fkps: ob[13:], acs: ac})
            ob, ac = loader.get_next_batch(-1, False)
            sess.run(loss, feed_dict={cfg: ob[:, :5], g: ob[:, 5:10],
                pobs: ob[:, 10:13], fkps: ob[13:], acs: ac})


train(1e-3, 1e-5, 20, 256)
