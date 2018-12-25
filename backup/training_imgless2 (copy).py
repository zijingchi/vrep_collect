import numpy as np
# import math
import keras
import os
# import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import BatchNormalization, Dropout, Dense, Input
from keras.callbacks import TensorBoard
from keras import optimizers, losses
from processing.DataGenerator import CustomDataGenWthTarCfg

learning_rate = 8e-3         # 学习率
# learning_rate = 0.1
lr_decay = 2e-1


def model_with_config_n_target():
    config = Input(shape=(6,), name='angles')
    target = Input(shape=(6,), name='target')
    obstacle_pos = Input(shape=(3,), name='obstacle_pos')
    obstacle_ori = Input(shape=(3,), name='obstacle_ori')
    merge1 = keras.layers.concatenate([config, target], name='merge1')
    merge1 = BatchNormalization(name='BN1')(merge1)
    x = Dense(64, activation='sigmoid', name='dense1')(merge1)
    x = Dense(32, activation='relu', name='dense2')(x)
    x = Dense(16, activation='relu', name='dense3')(x)
    #x = Dense(32, activation='relu', name='dense4')(x)
    #x = Dense(6, activation='relu', name='dense5')(x)
    y = BatchNormalization(name='BN2')(obstacle_pos)
    y = Dense(64, activation='sigmoid', name='obs_pos_dense1')(y)
    y = Dense(32, activation='relu', name='obs_pos_dense2')(y)
    y = Dense(16, activation='relu', name='obs_pos_dense3')(y)
    z = BatchNormalization(name='BN3')(obstacle_ori)
    z = Dense(64, activation='sigmoid', name='obs_ori_dense1')(z)
    z = Dense(32, activation='relu', name='obs_ori_dense2')(z)
    z = Dense(16, activation='relu', name='obs_ori_dense3')(z)
    merge2 = keras.layers.concatenate([y, z], name='obstacle')
    #merge2 = BatchNormalization(name='BN4')(merge2)
    m = Dense(64, activation='sigmoid', name='obs_dense1')(merge2)
    m = Dense(32, activation='relu', name='obs_dense2')(m)
    m = Dense(16, activation='relu', name='obs_dense3')(m)
    #m = Dense(6, activation='relu', name='obs_dense4')(m)
    merge3 = keras.layers.concatenate([x, m], name='merge3')
    final = Dense(128, activation='relu', name='final1')(merge3)
    final = Dense(64, activation='relu', name='final2')(final)
    final = Dense(32, activation='relu', name='final3')(final)
    final = Dense(16, activation='relu', name='final4')(final)
    final_output = Dense(6, name='final_output')(final)
    model = Model(inputs=[config, target, obstacle_pos, obstacle_ori],
                  outputs=final_output)
    model.compile(loss='mse',
                  optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=lr_decay),
                  metrics=['mae'])
    return model


def train_with_generator(datapath, batch_size, epochs):
    # model = model_with_config_n_target()
    h5file = '../train/h5files/test_with_target_obstacle_6.h5'
    model = load_model(h5file)
    model.compile(loss='mae',
                  optimizer=optimizers.Adam(lr=4e-4, beta_1=0.9, beta_2=0.999, decay=0.5),
                  metrics=['mae'])
    dirlist = os.listdir(datapath)
    id_list = []
    for d in dirlist:
        for i in range(50):
            id_list.append(d + '-' + str(i))
    id_size = len(id_list)
    train_size = int(0.7 * id_size)
    np.random.shuffle(id_list)
    train_list = id_list[:train_size]
    vali_list = id_list[train_size:]
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
                        callbacks=[TensorBoard(log_dir='./tensorboard_logs/target_obstacle_7/log')],
                        workers=2)
    model.save('./h5files/test_with_target_obstacle_7.h5')


if __name__ == '__main__':
    train_with_generator('/home/czj/vrep_path_dataset/4/', 50, 10)
