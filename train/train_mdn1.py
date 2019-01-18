import numpy as np
# import math
# import keras.backend as K
import os
# import tensorflow as tf
import pickle
from keras.models import Model, load_model, model_from_json
from keras.layers import BatchNormalization, Activation, Dense, Input, Dropout, Multiply
from keras.layers.merge import Subtract, Concatenate
from keras.callbacks import TensorBoard, TerminateOnNaN
from keras.utils import plot_model
from keras import optimizers, losses, regularizers
from processing.DataGenerator import CustomDataGenWthTarCfg
from train.mdn import *
from train.training_imgless2 import weighted_logcosh, model_with_config_n_target3

l1_regu = 1e-16
N_MIXES = 10
OUTPUT_DIMS = 5


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
    config = Input(shape=(dof,), name='angles')
    target = Input(shape=(dof,), name='target')
    obstacle = Input(shape=(24,), name='obstacle')

    big_model = model_with_config_n_target3(dof)
    final = big_model([config, target, obstacle])

    final_output = MDN(OUTPUT_DIMS, N_MIXES)(final)

    model = Model(inputs=[config, target, obstacle],
                  outputs=final_output)
    """model.compile(loss=weighted_logcosh,
                  optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=lr_decay),
                  metrics=['mse'])"""
    return model


def train_with_generator(datapath, batch_size, epochs):
    learning_rate = 2e-4  # 学习率
    lr_decay = 1e-3
    model = model_with_latentspace_mdn(5)
    model.load_weights('./h5files/5dof_latent_mdn_weights4.h5')
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
                                  callbacks=[TensorBoard(log_dir='./tensorboard_logs/5dof_latent_mdn2/log'),
                                             TerminateOnNaN()],
                                  workers=3)
    # K.clear_session()
    # model.save('./h5files/5dof_latent_6.h5')
    model.save_weights('./h5files/5dof_latent_mdn_weights5.h5')


if __name__ == '__main__':
    datapath = '/home/ubuntu/vdp/3/'
    train_with_generator(datapath, 100, 100)
