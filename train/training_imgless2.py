import numpy as np
# import math
import keras.backend as K
import os
import tensorflow as tf
import pickle
from keras.models import Model, load_model
from keras.layers import BatchNormalization, Activation, Dense, Input
from keras.layers.merge import Subtract, Concatenate
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras import optimizers, losses, regularizers
from keras.layers.core import Lambda
from processing.DataGenerator import CustomDataGenWthTarCfg
from processing.angle_dis import metrics
from train.fknodes import fktensor

learning_rate = 1e-2         # 学习率
# learning_rate = 0.1
lr_decay = 4e-3
l1_regul = 4e-5


def fklayer(batch_size):
    def fknode(x):
        output_tensor = []
        for i in range(batch_size):
            ps = fktensor(x[i])
            output_tensor.append(ps)
        output_tensor = tf.convert_to_tensor(output_tensor)
        return output_tensor
    return fknode


def model_with_config_n_target(dof, batch_size):
    """config = Input(shape=(dof,), name='angles')
    target = Input(shape=(dof,), name='target')
    obstacle = Input(shape=(24, ), name='obstacle')"""
    config = Input(batch_shape=(batch_size, dof), name='angles')
    target = Input(batch_shape=(batch_size, dof), name='target')
    obstacle = Input(batch_shape=(batch_size, 24), name='obstacle')

    fkn = Lambda(fklayer(batch_size), name='fklayer')(config)
    x1 = Dense(256, name='x1-dense1')(fkn)
    x1 = Dense(256, name='x1-dense2')(x1)
    x1 = BatchNormalization(name='x1-bn1')(x1)
    x1 = Activation('relu', name='x1-relu1')(x1)
    x1 = Dense(128, activation='relu',
               #kernel_regularizer=regularizers.l1(l1_regul),
               name='x1-dense3')(x1)
    x1 = Dense(64, activation='relu',
               #kernel_regularizer=regularizers.l1(l1_regul),
               name='x1-dense4')(x1)

    x2 = Dense(256, name='x2-dense1')(obstacle)
    x2 = Dense(256, name='x2-dense2')(x2)
    x2 = BatchNormalization(name='x2-bn1')(x2)
    x2 = Activation('relu', name='x2-relu2')(x2)
    x2 = Dense(128, activation='relu',
               #kernel_regularizer=regularizers.l1(l1_regul),
               name='x2-dense3')(x2)
    x2 = Dense(64, activation='relu',
               #kernel_regularizer=regularizers.l1(l1_regul),
               name='x2-dense4')(x2)

    merge1 = Concatenate(name='merge1', axis=1)([x1, x2])
    merge1 = Dense(256,
                   kernel_regularizer=regularizers.l1(l1_regul),
                   name='merge1-dense1')(merge1)
    merge1 = BatchNormalization(name='merge1-bn1')(merge1)
    merge1 = Activation('relu', name='merge1-relu1')(merge1)
    merge1 = Dense(128, activation='relu',
                   name='merge1-dense2')(merge1)
    merge1 = Dense(64, activation='relu', name='merge1-dense3')(merge1)

    x3 = Dense(512, name='config-dense1')(config)
    x3 = BatchNormalization(name='config-bn1')(x3)
    x3 = Activation(activation='relu', name='config-relu1')(x3)
    x3 = Dense(256, activation='relu', name='config-dense2')(x3)
    x3 = Dense(128, activation='relu', name='config-dense3')(x3)

    x4 = Dense(512, name='target-dense1')(target)
    x4 = BatchNormalization(name='target-bn1')(x4)
    x4 = Activation(activation='relu', name='target-relu1')(x4)
    x4 = Dense(256, activation='relu', name='target-dense2')(x4)
    x4 = Dense(128, activation='relu', name='target-dense3')(x4)

    merge2 = Concatenate(name='merge2', axis=1)([x3, x4])
    merge2 = Dense(256,
                   kernel_regularizer=regularizers.l1(l1_regul),
                   name='merge2-dense1')(merge2)
    merge2 = BatchNormalization(name='merge2-bn1')(merge2)
    merge2 = Activation('relu', name='merge2-relu1')(merge2)
    merge2 = Dense(128, activation='relu',
                   #kernel_regularizer=regularizers.l1(l1_regul),
                   name='merge2-dense2')(merge2)
    merge2 = Dense(64, activation='relu',
                   #kernel_regularizer=regularizers.l1(l1_regul),
                   name='merge2-dense3')(merge2)

    merge3 = Concatenate(name='merge3')([merge1, merge2])
    merge3 = Dense(256,
                   kernel_regularizer=regularizers.l1(l1_regul),
                   name='merge3-dense1')(merge3)
    merge3 = BatchNormalization(name='merge3-bn1')(merge3)
    merge3 = Activation('relu', name='merge3-relu1')(merge3)
    merge3 = Dense(128, activation='relu',
                   #kernel_regularizer=regularizers.l1(l1_regul),
                   name='merge3-dense2')(merge3)
    merge3 = Dense(64, activation='relu', name='merge3-dense3')(merge3)
    action = Dense(dof, name='final-output')(merge3)
    model = Model(inputs=[config, target, obstacle],
                  outputs=action)
    model.compile(loss=weighted_logcosh,
                  optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=lr_decay),
                  metrics=['mse'])
    return model


def weighted_logcosh(y_true, y_pred):

    def _logcosh(x):
        return x + K.softplus(-2. * x) - K.log(2.)

    tfmetrics = tf.convert_to_tensor(metrics, dtype=tf.float32)
    sub = tf.subtract(y_pred, y_true)
    weighted_sub = 5*tf.multiply(tfmetrics, sub)
    return K.mean(_logcosh(weighted_sub), axis=-1)


def separate_train_test(datapath):
    dirlist = os.listdir(datapath)
    id_list = []
    for d in dirlist:
        if d[0] == 'd':
            continue
        if os.path.isdir(os.path.join(datapath, d)):
            for i in range(50):
                id_list.append(d + '-' + str(i))
    id_size = len(id_list)
    train_size = int(0.8 * id_size)
    np.random.shuffle(id_list)
    train_list = id_list[:train_size]
    vali_list = id_list[train_size:]
    with open(os.path.join(datapath, 'list0.pkl'), 'wb') as f:
        pickle.dump({'train': train_list, 'test': vali_list}, f)


def separate_train_test2(datapath, listpkl, listpkl0=None):
    dirlist = os.listdir(datapath)
    id_list = []
    for d in dirlist:
        subdir = os.path.join(datapath, d)
        if os.path.isdir(subdir):
            datapkl = os.path.join(subdir, 'data.pkl')
            if os.path.exists(datapkl):
                with open(datapkl, 'rb') as dataf:
                    data = pickle.load(dataf)
                    for i in range(len(data['actions'])):
                        id_list.append(d + '-' + str(i))
    id_size = len(id_list)
    train_size = int(0.8 * id_size)
    np.random.shuffle(id_list)
    train_list = id_list[:train_size]
    vali_list = id_list[train_size:]
    if listpkl0:
        with open(os.path.join(datapath, listpkl0), 'rb') as f0:
            list0 = pickle.load(f0)
            train_list = train_list + list0['train']
            vali_list = vali_list + list0['test']
    with open(os.path.join(datapath, listpkl), 'wb') as f1:
        pickle.dump({'train': train_list, 'test': vali_list}, f1)


def train_with_generator(datapath, batch_size, epochs):
    model = model_with_config_n_target(5, batch_size)
    #model.load_weights('./h5files/model10_9_weights.h5')
    #model.compile(loss=weighted_logcosh,
    #              optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=lr_decay),
    #              metrics=['mse'])
    #model.summary()
    print(learning_rate, lr_decay, l1_regul)
    #h5file = './h5files/model10_2.h5'
    #model = load_model(h5file)
    """model.compile(loss=losses.logcosh,
              optimizer=optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, decay=0.05),
              metrics=['mae'])"""
    #plot_model(model, to_file='model10_1.jpg', show_shapes=True)
    with open(os.path.join(datapath, 'list1.pkl'), 'rb') as f:
        lists = pickle.load(f)
        train_list = lists['train']
        vali_list = lists['test']

    train_gen = CustomDataGenWthTarCfg(datapath=datapath,
                                       list_IDs=train_list,
                                       data_size=50,
                                       batch_size=batch_size)
    vali_gen = CustomDataGenWthTarCfg(datapath=datapath,
                                      list_IDs=vali_list,
                                      data_size=50,
                                      batch_size=batch_size)
    history = model.fit_generator(generator=train_gen,
                                  epochs=epochs,
                                  validation_data=vali_gen,
                                  use_multiprocessing=True,
                                  callbacks=[TensorBoard(log_dir='./tensorboard_logs/model10_5/log')],
                                  workers=3)
    # K.clear_session()
    #model.save('./h5files/model10_5.h5')
    model.save_weights('./h5files/model10_10_weights.h5')


if __name__ == '__main__':
    datapath = '/home/ubuntu/vrep_path_dataset/2/'
    #separate_train_test2(datapath, 'list1.pkl', 'list0.pkl')
    train_with_generator(datapath, 72, 400)
