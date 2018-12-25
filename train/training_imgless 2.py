import numpy as np
# import math
# import keras.backend as K
import os
# import tensorflow as tf
import pickle
from keras.models import Model, load_model
from keras.layers import BatchNormalization, Activation, Dense, Input
from keras.layers.merge import Subtract, Concatenate
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras import optimizers, losses
from processing.DataGenerator import CustomDataGenWthTarCfg

learning_rate = 1e-2         # 学习率
# learning_rate = 0.1
lr_decay = 1e-1


def model_with_config_n_target():
    config = Input(shape=(6,), name='angles')
    target = Input(shape=(6,), name='target')
    sub = Subtract(name='subtract')([target, config])
    obstacle_posnori = Input(shape=(6,), name='obstacle_pos')
    x1 = Dense(512, name='config_dense1')(config)
    x1 = BatchNormalization(name='config_bn')(x1)
    x1 = Activation('relu', name='config_relu1')(x1)
    x1 = Dense(256, activation='relu', name='config_dense2')(x1)
    x1 = Dense(128, activation='relu', name='config_dense3')(x1)
    x2 = Dense(512, name='target_dense1')(target)
    x2 = BatchNormalization(name='target_bn')(x2)
    x2 = Activation('relu', name='target_relu1')(x2)
    x2 = Dense(256, activation='relu', name='target_dense2')(x2)
    x2 = Dense(128, activation='relu', name='target_dense3')(x2)
    x3 = Dense(128, activation='relu', name='sub_dense1')(sub)
    x3 = Dense(64, activation='relu', name='sub_dense2')(x3)
    x3 = Dense(32, activation='relu', name='sub_dnese3')(x3)
    merge1 = Concatenate(name='merge_config_target')([x1, x2])
    x = Dense(256, activation='relu', name='merge_dense1')(merge1)
    x = Dense(128, activation='relu', name='merge_dense2')(x)

    y = Dense(512, name='obstacle_dense1')(obstacle_posnori)
    #y = BatchNormalization(name='obstacle_bn')(y)
    y = Activation('relu', name='obstacle_relu1')(y)
    y = Dense(256, activation='relu', name='obstacle_dense2')(y)
    y = Dense(128, activation='relu', name='obstacle_dense3')(y)

    merge2 = Concatenate(name='merge_config_obstacle')([x, y])
    final = Dense(512, name='final_dense1')(merge2)
    #final = BatchNormalization(name='final_bn')(final)
    final = Activation('relu', name='final_relu')(final)
    final = Dense(256, activation='relu', name='final_dense2')(final)
    # final = Dense(64, activation='relu', name='final_dense3')(final)
    final = Dense(32, activation='relu', name='final_dense4')(final)
    final = Concatenate(name='final_merge')([final, x3])
    final_output = Dense(6, name='final_output')(final)
    model = Model(inputs=[config, target, obstacle_posnori],
                  outputs=final_output)
    model.compile(loss=losses.logcosh,
                  optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=lr_decay),
                  metrics=['mae'])
    return model


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
    with open(os.path.join(datapath, 'list2.pkl'), 'rb') as f0:
        list0 = pickle.load(f0)
        train_list = train_list + list0['train']
        vali_list = vali_list + list0['test']
    with open(os.path.join(datapath, 'list3.pkl'), 'wb') as f1:
        pickle.dump({'train': train_list, 'test': vali_list}, f1)


def train_with_generator(datapath, batch_size, epochs):
    # model = model_with_config_n_target()
    h5file = './h5files/model7_dagger16.h5'
    model = load_model(h5file)
    model.compile(loss=losses.logcosh,
                  optimizer=optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, decay=0.05),
                  metrics=['mae'])
    # plot_model(model, to_file='model7.jpg', show_shapes=True)
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
                        callbacks=[TensorBoard(log_dir='./tensorboard_logs/model7_d4/log')],
                        workers=3)
    # K.clear_session()
    model.save('./h5files/model7_dagger17.h5')


if __name__ == '__main__':
    datapath = '/home/czj/vrep_path_dataset/10/'
    #separate_train_test2(datapath)
    train_with_generator(datapath, 64, 80)
