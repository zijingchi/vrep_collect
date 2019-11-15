#from keras import backend as K
import tensorflow as tf
#from keras.layers.core import Lambda
import numpy as np
pi = np.pi


def fktensor(x):
    d0 = 0.3
    A0 = [[0., 1., 0., 0.], [-1., 0., 0., 0.], [0., 0., 1., d0], [0., 0., 0., 1.]]
    A0 = tf.convert_to_tensor(A0)
    d1 = 8.92e-2
    d2 = 0.11
    d5 = 9.475e-2
    d6 = 7.495e-2
    a2 = 4.251e-1
    a3 = 3.9215e-1
    A1 = urfka1(x[0], d1)
    A2 = urfka2(x[1], d2)
    A3 = urfka3(x[2], a2)
    A4 = urfka4(x[3], a3)
    A5 = urfka5(x[4], d5)
    Q1 = tf.matmul(A0, A1)
    #t1 = Q1[:3, 3]
    Q2 = tf.matmul(Q1, A2)
    #t2 = Q2[:3, 3]
    Q3 = tf.matmul(Q2, A3)
    #t3 = Q3[:3, 3]
    Q4 = tf.matmul(Q3, A4)
    #t4 = Q4[:3, 3]
    #A34 = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., -0.1], [0., 0., 0., 1.]]
    #A34 = tf.convert_to_tensor(A34)
    """A34 = tf.SparseTensor(indices=[[0, 0], [1, 1], [2, 2], [3, 3], [2, 3]],
                          values=[1., 1., 1., 1., -0.1],
                          dense_shape=[4, 4])"""
    #Q34 = tf.matmul(Q4, A34)
    #t34 = Q34[:3, 3]
    Q5 = tf.matmul(Q4, A5)
    #t5 = Q5[:3, 3]
    t6 = tf.constant([[0.], [-d6], [0.], [1.]])
    Q6 = tf.matmul(Q5, t6)
    t6 = Q6[:3, 0]
    #ps = tf.concat([t2, t3, t34, t4, t5, t6], axis=0)
    return t6


def urfka1(theta1, d1):
    a1 = tf.cos(theta1)
    a2 = tf.sin(theta1)
    a3 = tf.negative(a2)
    a5 = tf.constant(1.)
    a6 = tf.constant(d1)
    #A1 = tf.zeros((4, 4))
    A1 = [[a1, a3, 0., 0.], [a2, a1, 0., 0.], [0., 0., a5, a6], [0., 0., 0., a5]]
    A1 = tf.convert_to_tensor(A1)
    """A1 = tf.SparseTensor(indices=[[0, 0], [1, 0], [0, 1], [1, 1], [2, 2], [2, 3], [3, 3]],
                         values=[a1, a2, a3, a1, a5, a6, a5],
                         #values=[1, 1, 1, 1, 1, 1, 1],
                         dense_shape=[4, 4])"""
    return A1


def urfka2(theta2, d2):
    c1 = tf.negative(tf.sin(theta2))
    c2 = tf.cos(theta2)
    c3 = tf.negative(c2)
    c4 = tf.constant(-1.)
    c5 = tf.constant(-d2)
    c6 = tf.constant(1.)
    A2 = [[c1, c3, 0., 0.], [0., 0., c4, c5], [c2, c1, 0., 0.], [0., 0., 0., c6]]
    A2 = tf.convert_to_tensor(A2)
    """A2 = tf.SparseTensor(indices=[[0, 0], [2, 0], [0, 1], [2, 1], [1, 2], [1, 3], [3, 3]],
                         values=[c1, c2, c3, c1, c4, c5, c6],
                         dense_shape=[4, 4])"""
    return A2


def urfka3(theta3, a):
    a1 = tf.cos(theta3)
    a2 = tf.sin(theta3)
    a3 = tf.negative(a2)
    a5 = tf.constant(1.)
    a6 = tf.constant(a)
    A3 = [[a1, a3, 0., a6], [a2, a1, 0., 0.], [0., 0., a5, 0.], [0., 0., 0., 1.]]
    A3 = tf.convert_to_tensor(A3)
    """A3 = tf.SparseTensor(indices=[[0, 0], [1, 0], [0, 1], [1, 1], [2, 2], [0, 3], [3, 3]],
                         values=[a1, a2, a3, a1, a5, a6, a5],
                         dense_shape=[4, 4])"""
    return A3


def urfka4(theta4, a):
    a1 = tf.negative(tf.sin(theta4))
    a2 = tf.cos(theta4)
    a3 = tf.negative(a2)
    a5 = tf.constant(1.0)
    a6 = tf.constant(a)
    A4 = [[a1, a3, 0., a6], [a2, a1, 0., 0.], [0., 0., a5, 0], [0., 0., 0., a5]]
    A4 = tf.convert_to_tensor(A4)
    """A3 = tf.SparseTensor(indices=[[0, 0], [1, 0], [0, 1], [1, 1], [2, 2], [0, 3], [3, 3]],
                         values=[a1, a2, a3, a1, a5, a6, a5],
                         dense_shape=[4, 4])"""
    return A4


def urfka5(theta5, d5):
    a1 = tf.negative(tf.cos(theta5))
    a2 = tf.sin(theta5)
    a3 = tf.negative(a2)
    a5 = tf.constant(-1.0)
    a6 = tf.constant(-d5)
    a7 = tf.constant(1.0)
    A5 = [[a1, a2, 0., 0.], [0., 0., a5, a6], [a3, a1, 0., 0.], [0., 0., 0., a7]]
    """A5 = tf.SparseTensor(indices=[[0, 0], [0, 1], [2, 0], [2, 1], [1, 2], [1, 3], [3, 3]],
                         values=[a1, a2, a3, a1, a5, a6, a7],
                         dense_shape=[4, 4])"""
    return A5


def ur5fk(thetas):
    d1 = 8.92e-2
    d2 = 0.11
    d5 = 9.475e-2
    d6 = 7.495e-2
    a2 = 4.251e-1
    a3 = 3.9215e-1
    d0 = 0.3
    All = np.zeros((6, 4, 4))
    All[:, 3, 3] = 1
    for i in range(5):
        All[i, 0, 0] = np.cos(thetas[i])
        All[i, 0, 1] = -np.sin(thetas[i])
    All[0, 1, 0] = np.sin(thetas[0])
    All[0, 1, 1] = np.cos(thetas[0])
    All[0, 2, 3] = d1
    All[0, 2, 2] = 1

    All[1, 2, 0] = np.sin(thetas[1])
    All[1, 2, 1] = np.cos(thetas[1])
    All[1, 1, 2] = -1
    All[1, 1, 3] = -d2

    All[2, 1, 0] = np.sin(thetas[2])
    All[2, 1, 1] = np.cos(thetas[2])
    All[2, 0, 3] = a2
    All[2, 2, 2] = 1

    All[3, 1, 0] = np.sin(thetas[3])
    All[3, 1, 1] = np.cos(thetas[3])
    All[3, 0, 3] = a3
    All[3, 2, 2] = 1

    All[4, 2, 0] = np.sin(thetas[4])
    All[4, 2, 1] = np.cos(thetas[4])
    All[4, 1, 3] = -d5
    All[4, 1, 2] = -1

    All[5, :, :] = np.eye(4)
    All[5, 1, 3] = -d6

    A0 = np.zeros((4, 4))
    A0[0, 1] = 1
    A0[1, 0] = -1
    A0[2, 2] = 1
    A0[3, 3] = 1
    A0[2, 3] = d0
    return All, A0


def tipcoor(thetas):
    thetas_0 = np.array([0, pi / 2, 0, pi / 2, pi])
    thetas = thetas + thetas_0
    All, A0 = ur5fk(thetas)
    ps = []
    for i, A in enumerate(All):
        A0 = A0 @ A
        ps.extend(A0[:3, 3])
        if i==2 or i==3:
            p = A0[:3, 3]-0.1*A0[:3, 2]
            ps.extend(p)
    return np.array(ps)


def fklayer(x, n):
    output = []
    for i in range(n):
        y = fktensor(x[i])
        output.append(y)
    return tf.convert_to_tensor(output)

def main():
    """thetas_0 = np.array([0, pi/2, 0, pi/2, pi, 0])
    thetas = np.array([0, 0, 0, 0, 0, 0]) + thetas_0
    All, A0 = ur5fk(thetas)
    #A0 = np.eye(4)
    for A in All:
        A0 = A0 @ A
    print(A0[:3, 3])"""
    thetas = tf.placeholder(tf.float32, shape=(2, 5), name='theta')
    #thetas = tf.constant([0., 0., 0., 0., 0.])
    ps = fklayer(thetas, 2)
    with tf.Session() as sess:
        print(sess.run(ps, feed_dict={thetas: [[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]]}))


if __name__ == '__main__':
    main()
