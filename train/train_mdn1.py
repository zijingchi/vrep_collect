import numpy as np
# import math
# import keras.backend as K
import os
# import tensorflow as tf
import pickle
from keras.models import Model, load_model, model_from_json
from keras.layers import BatchNormalization, Activation, Dense, Input, Dropout, Multiply, Add
from keras.layers.merge import Subtract, Concatenate
from keras.callbacks import TensorBoard, TerminateOnNaN, ModelCheckpoint
from keras.utils import plot_model
from keras import optimizers, losses, regularizers
from processing.DataGenerator import CustomDataGenWthTarCfg
from train.mdn import *
from train.training_imgless2 import weighted_logcosh, model_with_1dconv
from train.training_imgless import model_with_config_n_target2

l1_regu = 1e-16
N_MIXES = 10
OUTPUT_DIMS = 3


def fdlp4theta(dof):
    """map the theta to a latent space"""
    input = Input(shape=(dof,))
    y = Dense(64,
              kernel_regularizer=regularizers.l1(l1_regu),
              bias_regularizer=regularizers.l1(l1_regu),
              name='f-dense1')(input)
    y = BatchNormalization(name='f-bn1')(y)
    y = Activation('relu', name='f-relu1')(y)
    y = Dense(64, activation='relu', name='f-dense2')(y)
    y = Dense(64, name='f-dense3')(y)
    y = BatchNormalization(name='f-bn2')(y)
    y = Activation('relu', name='f-relu2')(y)
    #y = Dropout(0.5, name='f-dp1')(y)
    y = Dense(32, activation='relu', name='f-dense4')(y)
    fmodel = Model(inputs=input, outputs=y)
    return fmodel


def model_with_latentspace_mdn(dof):
    config = Input(shape=(5,), name='angles')
    target = Input(shape=(5,), name='target')
    #obstacle = Input(shape=(8, 3, ), name='obstacle')
    obstacle_pos = Input(shape=(3,), name='obs-pos')
    obstacle_ori = Input(shape=(3,), name='obs-ori')

    big_model = model_with_config_n_target2(dof)
    act1 = big_model([config, target, obstacle_pos, obstacle_ori])

    act1_param = MDN(OUTPUT_DIMS, N_MIXES)(act1)
    final_output = act1_param

    model = Model(inputs=[config, target, obstacle_pos, obstacle_ori],
                  outputs=final_output)
    """model.compile(loss=weighted_logcosh,
                  optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=lr_decay),
                  metrics=['mse'])"""
    return model


def train_with_generator(datapath, batch_size, epochs, dof):
    learning_rate = 8e-4  # 学习率
    lr_decay = 3e-3
    model = model_with_latentspace_mdn(dof)
    model.load_weights('./h5files/mdn_weight_4_7_6_check.h5')
    model.compile(loss=get_mixture_loss_func(OUTPUT_DIMS, N_MIXES),
                  optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=lr_decay),
                  metrics=[get_mixture_mse_accuracy(OUTPUT_DIMS, N_MIXES)])
    # model.summary()
    # print(learning_rate, lr_decay)
    # plot_model(model, to_file='5dof_latent3.jpg', show_shapes=True)
    with open(os.path.join(datapath, 'list0.pkl'), 'rb') as f:
        lists = pickle.load(f)
        train_list = lists['train']
        vali_list = lists['test']

    train_gen = CustomDataGenWthTarCfg(datapath=datapath,
                                       list_IDs=train_list,
                                       data_size=dof,
                                       batch_size=batch_size)
    vali_gen = CustomDataGenWthTarCfg(datapath=datapath,
                                      list_IDs=vali_list,
                                      data_size=dof,
                                      batch_size=batch_size)
    checkpoint = ModelCheckpoint(filepath='./h5files/mdn_weight_4_7_8_check.h5', monitor='val_mse_func',
                                 save_best_only=True, save_weights_only=True, mode='min')
    history = model.fit_generator(generator=train_gen,
                                  epochs=epochs,
                                  validation_data=vali_gen,
                                  use_multiprocessing=True,
                                  callbacks=[TensorBoard(log_dir='./tensorboard_logs/mdn_4_7_8/log'),
                                             TerminateOnNaN(),
                                             checkpoint],
                                  workers=2)
    # K.clear_session()
    # model.save('./h5files/5dof_latent_6.h5')
    model.save_weights('./h5files/mdn_4_7_8.h5')


if __name__ == '__main__':
    datapath = '/home/ubuntu/vdp/4_7/'
    train_with_generator(datapath, 72, 200, OUTPUT_DIMS)
