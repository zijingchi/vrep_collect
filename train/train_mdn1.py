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
from train.train3 import weighted_logcosh

l1_regu = 1e-5
N_MIXES = 30
OUTPUT_DIMS = 5


def model_with_config_n_target():
    config = Input(shape=(5,), name='angles')
    target = Input(shape=(5,), name='target')
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
    y = Dense(128, activation='relu', name='final_dense3')(y)
    final_output = MDN(OUTPUT_DIMS, N_MIXES)(y)
    model = Model(inputs=[config, target, obstacle_posnori],
                  outputs=final_output)
    """model.compile(loss=get_mixture_loss_func(OUTPUT_DIMS, N_MIXES),
                  optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=lr_decay),
                  metrics=[get_mixture_mse_accuracy(OUTPUT_DIMS, N_MIXES)])"""
    # model.summary()
    return model


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

    x1 = Dense(64, name='config-dense1',
               kernel_regularizer=regularizers.l1(l1_regu),
               #bias_regularizer=regularizers.l2(l2_regu)
               )(config)
    x1 = BatchNormalization(name='config-bn1')(x1)
    x1 = Activation('relu', name='config-relu1')(x1)
    x1 = Dense(32, activation='relu', name='config-dense2')(x1)
    x2 = Dense(64, name='target-dense1',
               kernel_regularizer=regularizers.l1(l1_regu),
               #bias_regularizer=regularizers.l2(l2_regu)
               )(target)
    x2 = BatchNormalization(name='target-bn1')(x2)
    x2 = Activation('relu', name='target-relu1')(x2)
    x2 = Dense(32, activation='relu', name='target-dense2')(x2)
    x3 = Dense(64, name='obs-dense1',
               kernel_regularizer=regularizers.l1(l1_regu),
               #bias_regularizer=regularizers.l2(l2_regu)
               )(obstacle)
    x3 = BatchNormalization(name='obs-bn1')(x3)
    x3 = Activation('relu', name='obs-relu1')(x3)
    x3 = Dense(32, activation='relu', name='obs-dense2')(x3)

    merge1 = Concatenate(name='concat')([x1, x2, x3])
    alpha = Dense(64, name='alpha-dense1')(merge1)
    alpha = BatchNormalization(name='alpha-bn1')(alpha)
    alpha = Activation('relu', name='alpha-relu1')(alpha)
    alpha = Dense(64, activation='relu',
                  kernel_regularizer=regularizers.l1(l1_regu),
                  bias_regularizer=regularizers.l1(l1_regu),
                  name='alpha-dense2')(alpha)
    alpha = Dropout(0.5, name='alpha-dp1')(alpha)
    alpha = Dense(5, name='alpha-final')(alpha)

    beta = Dense(64, name='beta-dense1')(merge1)
    beta = BatchNormalization(name='beta-bn1')(beta)
    beta = Activation('relu', name='beta-relu1')(beta)
    beta = Dense(64, activation='relu',
                 kernel_regularizer=regularizers.l1(l1_regu),
                 bias_regularizer=regularizers.l1(l1_regu),
                 name='beta-dense2')(beta)
    beta = Dropout(0.5, name='beta-dp1')(beta)
    beta = Dense(32, name='beta-final')(beta)

    theta_sub = Subtract(name='target-config')([target, config])
    multi1 = Multiply(name='alpha_sub')([alpha, theta_sub])

    o = Dense(64,
              kernel_regularizer=regularizers.l1(l1_regu),
              bias_regularizer=regularizers.l1(l1_regu),
              name='obs-latent-dense1')(obstacle)
    o = BatchNormalization(name='obs-latent-bn1')(o)
    o = Activation('relu', name='obs-latent-relu1')(o)
    o = Dense(64, activation='relu', name='obs-latent-dense2')(o)
    o = Dense(32, activation='relu', name='obs-latent-dense3')(o)
    t = Dense(64, name='target-obs-dense1',
              kernel_regularizer=regularizers.l1(l1_regu),
              bias_regularizer=regularizers.l1(l1_regu))(target)
    t = BatchNormalization(name='target-obs-bn1')(t)
    t = Activation('relu', name='target-obs-relu')(t)
    t = Dense(64, activation='relu', name='target-obs-dense2')(t)
    t = Dense(32, activation='relu', name='target-obs-dense3')(t)
    obs_all = Concatenate(name='obs-merge')([o, t])
    obs_all = Dense(64, activation='relu',
                    name='obs-latent-dense4',
                    kernel_regularizer=regularizers.l1(l1_regu),
                    bias_regularizer=regularizers.l1(l1_regu))(obs_all)
    obs_all = Dense(64, activation='relu', name='obs-latent-dense5')(obs_all)
    obs_all = Dense(32, activation='relu', name='obs-latent-dense6')(obs_all)
    fmodel = fdlp4theta(dof)
    latent_config = fmodel(config)
    obs_sub = Subtract(name='config-obs')([latent_config, obs_all])
    obs_sub = Dense(128,
                    kernel_regularizer=regularizers.l1(l1_regu),
                    bias_regularizer=regularizers.l1(l1_regu),
                    name='obs-sub-dense1')(obs_sub)
    obs_sub = BatchNormalization(name='obs-sub-bn1')(obs_sub)
    obs_sub = Activation('relu', name='obs-sub-relu1')(obs_sub)
    obs_sub = Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l1(l1_regu),
                    bias_regularizer=regularizers.l1(l1_regu),
                    name='obs-sub-dense2')(obs_sub)
    obs_sub = Dense(32, activation='relu', name='obs-sub-dense3')(obs_sub)
    multi2 = Multiply(name='beta_sub')([beta, obs_sub])
    latent_action = Concatenate(name='theta_obs')([multi1, multi2])

    final = Dense(128, name='final-dense1',
                  kernel_regularizer=regularizers.l1(4*l1_regu))(latent_action)
    final = BatchNormalization(name='final-bn1')(final)
    final = Activation('relu', name='final-relu1')(final)
    final = Dense(64, activation='relu',
                  name='final-dense2',
                  kernel_regularizer=regularizers.l1(l1_regu),
                  bias_regularizer=regularizers.l1(l1_regu))(final)
    #final = Dropout(0.5, name='final-dp1')(final)
    final = Dense(32, activation='relu',
                  name='final-dense3',
                  kernel_regularizer=regularizers.l1(l1_regu),
                  bias_regularizer=regularizers.l1(l1_regu))(final)
    N_MIXES = 30
    OUTPUT_DIMS = 5
    final_output = MDN(OUTPUT_DIMS, N_MIXES)(final)

    model = Model(inputs=[config, target, obstacle],
                  outputs=final_output)
    """model.compile(loss=weighted_logcosh,
                  optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=lr_decay),
                  metrics=['mse'])"""
    return model


def loadmodeltest():
    model = model_with_config_n_target()
    model.load_weights('./h5files/model8_5_weights.h5')
    config = model.get_config()
    json_string = model.to_json()
    #with open('./model_json.pkl', 'wb') as f2:
    #    pickle.dump(json_string, f2)
    model2 = model_from_json(json_string)
    return model2


def train_with_generator(datapath, batch_size, epochs):
    learning_rate = 2e-2  # 学习率
    lr_decay = 1e-3
    model = model_with_latentspace_mdn(5)
    # model.load_weights('./h5files/5dof_latent_weights6.h5')
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
                                  callbacks=[TensorBoard(log_dir='./tensorboard_logs/5dof_latent_mdn/log'),
                                             TerminateOnNaN()],
                                  workers=3)
    # K.clear_session()
    # model.save('./h5files/5dof_latent_6.h5')
    model.save_weights('./h5files/5dof_latent_mdn_weights1.h5')


if __name__ == '__main__':
    datapath = '/home/czj/vrep_path_dataset/2_1/'
    train_with_generator(datapath, 64, 400)
