import numpy as np
# import math
import keras
import os
import pickle
# import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import BatchNormalization, Concatenate, Dense, Input, Activation, Subtract
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras import optimizers, losses
from processing.DataGenerator import CustomDataGenWthTarCfg

learning_rate = 1e-2        # 学习率
# learning_rate = 0.1
lr_decay = 1e-3


def model_with_config_n_target(dof):
    config = Input(shape=(dof,), name='angles')
    target = Input(shape=(dof,), name='target')
    obstacle_posnori = Input(shape=(6,), name='obstacle_pos')
    x1 = Dense(512, name='config_dense1')(config)
    x1 = BatchNormalization(name='config_bn1')(x1)
    x1 = Activation(activation='relu', name='config_relu1')(x1)
    x1 = Dense(256, name='config_dense2')(x1)
    x1 = BatchNormalization(name='config_bn2')(x1)
    x1 = Activation(activation='relu', name='config_relu2')(x1)
    x1 = Dense(128, activation='relu', name='config_dense3')(x1)
    x1 = Dense(64, activation='relu', name='config_dense4')(x1)

    x2 = Dense(512, name='obs_dense1')(obstacle_posnori)
    x2 = BatchNormalization(name='obs_bn1')(x2)
    x2 = Activation(activation='relu', name='obs_relu1')(x2)
    x2 = Dense(256, name='obs_dense2')(x2)
    x2 = BatchNormalization(name='obs_bn2')(x2)
    x2 = Activation(activation='relu', name='obs_relu2')(x2)
    x2 = Dense(128, activation='relu', name='obs_dense3')(x2)
    x2 = Dense(64, name='obs_dense4')(x2)
    # x2 = BatchNormalization(name='obs_bn2')(x2)
    x2 = Activation('relu', name='obs_relu4')(x2)

    x3 = Concatenate(name='configNobs')([x1, x2])
    x3 = Dense(128, name='co_dense1')(x3)
    x3 = Dense(64, name='co_dense2')(x3)
    x3 = BatchNormalization(name='co_bn1')(x3)
    x3 = Activation('relu', name='co_relu1')(x3)
    x3 = Dense(32, activation='relu', name='co_dense3')(x3)

    x4 = Subtract(name='sub')([target, config])
    x4 = Dense(128, name='sub_dense1')(x4)
    x4 = BatchNormalization(name='sub_bn1')(x4)
    x4 = Activation('relu', name='sub_relu1')(x4)
    x4 = Dense(32, activation='relu', name='sub_dense2')(x4)

    y = Concatenate(name='merge')([x3, x4])
    y = Dense(256, name='final_dense1')(y)
    y = BatchNormalization(name='final_bn1')(y)
    y = Activation('relu', name='final_relu1')(y)
    y = Dense(128, activation='relu', name='final_dense2')(y)
    y = Dense(32, activation='relu', name='final_dense3')(y)

    final_output = Dense(dof, name='final_output')(y)
    model = Model(inputs=[config, target, obstacle_posnori],
                  outputs=final_output)
    model.compile(loss=losses.logcosh,
                  optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=lr_decay),
                  metrics=['mae'])
    return model


def separate_train_test2(datapath):
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
    """with open(os.path.join(datapath, 'list0.pkl'), 'rb') as f0:
        list0 = pickle.load(f0)
        train_list = train_list + list0['train']
        vali_list = vali_list + list0['test']"""
    with open(os.path.join(datapath, 'list0.pkl'), 'wb') as f1:
        pickle.dump({'train': train_list, 'test': vali_list}, f1)


def train_with_generator(datapath, batch_size, epochs):
    #model = model_with_config_n_target(5)
    h5file = '../train/h5files/model6_5.h5'
    model = load_model(h5file)

    model.compile(loss=losses.logcosh,
                  optimizer=optimizers.Adam(lr=3e-4, beta_1=0.9, beta_2=0.999, decay=1e-3),
                  metrics=['mae'])
    # plot_model(model, to_file='model5.jpg', show_shapes=True)
    with open(os.path.join(datapath, 'list0.pkl'), 'rb') as f:
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
    model.fit_generator(generator=train_gen,
                        epochs=epochs,
                        validation_data=vali_gen,
                        use_multiprocessing=True,
                        callbacks=[TensorBoard(log_dir='./tensorboard_logs/model6_1/log')],
                        workers=2)
    model.save('./h5files/model6_6.h5')


if __name__ == '__main__':
    datapath = '/home/ubuntu/vrep_path_dataset/21/'
    #separate_train_test2(datapath)
    train_with_generator(datapath, 64, 400)
