import numpy as np
# import math
import keras
import os
import pickle
# import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import BatchNormalization, Dropout, Dense, Input
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras import optimizers, losses
from processing.DataGenerator import CustomDataGenWthTarCfg

learning_rate = 1e-3        # 学习率
# learning_rate = 0.1
lr_decay = 0.1


def model_with_config_n_target():
    config = Input(shape=(6,), name='angles')
    target = Input(shape=(6,), name='target')
    obstacle_posnori = Input(shape=(6,), name='obstacle_pos')
    x1 = Dense(512, activation='relu', name='config_dense1')(config)
    x1 = Dense(256, activation='relu', name='config_dense2')(x1)
    x1 = BatchNormalization(name='BN1')(x1)
    x1 = Dense(64, activation='relu', name='config_dense3')(x1)
    x2 = Dense(512, activation='relu', name='target_dense1')(target)
    x2 = Dense(256, activation='relu', name='target_dense2')(x2)
    x2 = BatchNormalization(name='BN2')(x2)
    x2 = Dense(64, activation='relu', name='target_dense3')(x2)
    merge1 = keras.layers.concatenate([x1, x2], name='merge_config_target')
    x = Dense(512, activation='relu', name='merge_dense1')(merge1)
    x = Dense(128, activation='relu', name='merge_dense2')(x)

    y = Dense(256, activation='relu', name='obstacle_dense1')(obstacle_posnori)
    y = Dense(256, activation='relu', name='obstacle_dense2')(y)
    y = BatchNormalization(name='BN3')(y)
    y = Dense(128, activation='relu', name='obstacle_dense3')(y)

    merge2 = keras.layers.concatenate([x, y], name='merge_config_obstacle')
    final = Dense(1024, activation='relu', name='final_dense1')(merge2)
    final = Dense(256, activation='relu', name='final_dense2')(final)
    final = Dense(64, activation='relu', name='final_dense3')(final)
    #final = Dense(32, activation='relu', name='final_dense4')(final)
    final_output = Dense(6, name='final_output')(final)
    model = Model(inputs=[config, target, obstacle_posnori],
                  outputs=final_output)
    model.compile(loss=losses.logcosh,
                  optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=lr_decay),
                  metrics=['mae'])
    return model


def train_with_generator(datapath, batch_size, epochs):
    #model = model_with_config_n_target()
    h5file = '../train/h5files/model5_1.h5'
    model = load_model(h5file)

    model.compile(loss=losses.logcosh,
                  optimizer=optimizers.Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, decay=0.1),
                  metrics=['mae'])
    plot_model(model, to_file='model5.jpg', show_shapes=True)
    dirlist = os.listdir(datapath)
    id_list = []
    for d in dirlist:
        subdir = os.path.join(datapath, d)
        pklfile = os.path.join(subdir, 'seleted_list.pkl')
        with open(pklfile, 'rb') as f:
            slist = pickle.load(f)
            for n in slist:
                id_list.append(d + '-' + str(n))
        #for i in range(50):
            #id_list.append(d + '-' + str(i))
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
                        callbacks=[TensorBoard(log_dir='./tensorboard_logs/model5/log')],
                        workers=2)
    model.save('./h5files/model5_2.h5')


if __name__ == '__main__':
    train_with_generator('/home/czj/vrep_path_dataset/4/', 64, 20)
