import numpy as np
# import math
import keras.backend as K
import os
import tensorflow as tf
import pickle
from keras.models import Model, load_model
from keras.layers import BatchNormalization, Activation, Dense, Input, Dropout
from keras.layers.merge import Subtract, Concatenate, Multiply, Add
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras import optimizers, losses, regularizers
from processing.DataGenerator import CustomDataGenWthTarCfg
from processing.angle_dis import metrics

l1_regu = 1e-10
l2_regu = 1e-10
#metrics = [1, 4, 3.2, 0.3, 0.2]


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


def model_with_latentspace(dof):
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
    x3 = Dense(64,
               kernel_regularizer=regularizers.l1(l1_regu),
               #bias_regularizer=regularizers.l2(l2_regu),
               name='obs-dense1')(obstacle)
    x3 = BatchNormalization(name='obs-bn1')(x3)
    x3 = Activation('relu', name='obs-relu1')(x3)
    x3 = Dense(32, activation='relu', name='obs-dense2')(x3)

    merge1 = Concatenate(name='concat')([x1, x2, x3])
    alpha = Dense(64, name='alpha-dense1')(merge1)
    alpha = BatchNormalization(name='alpha-bn1')(alpha)
    alpha = Activation('relu', name='alpha-relu1')(alpha)
    alpha = Dense(64, activation='relu', name='alpha-dense2',
                  kernel_regularizer=regularizers.l1(l1_regu),
                  bias_regularizer=regularizers.l1(l1_regu)
                  )(alpha)
    alpha = Dropout(0.5, name='alpha-dp1')(alpha)
    alpha = Dense(32, name='alpha-final')(alpha)

    beta = Dense(64, name='beta-dense1')(merge1)
    beta = BatchNormalization(name='beta-bn1')(beta)
    beta = Activation('relu', name='beta-relu1')(beta)
    beta = Dense(64, activation='relu', name='beta-dense2',
                 kernel_regularizer=regularizers.l1(l1_regu),
                 bias_regularizer=regularizers.l1(l1_regu)
                 )(beta)
    beta = Dropout(0.5, name='beta-dp1')(beta)
    beta = Dense(32, name='beta-final')(beta)

    fmodel = fdlp4theta(dof)
    latent_config = fmodel(config)
    latent_target = fmodel(target)
    theta_sub = Subtract(name='target-config')([latent_target, latent_config])
    multi1 = Multiply(name='alpha_sub')([alpha, theta_sub])

    o = Dense(128, name='obs-latent-dense1',
              kernel_regularizer=regularizers.l1(l1_regu),
              bias_regularizer=regularizers.l1(l1_regu))(obstacle)
    o = BatchNormalization(name='obs-latent-bn1')(o)
    o = Activation('relu', name='obs-latent-relu1')(o)
    o = Dense(64, activation='relu', name='obs-latent-dense2')(o)
    o = Dense(64, activation='relu', name='obs-latent-dense3')(o)
    t = Dense(64, name='target-obs-dense1',
              kernel_regularizer=regularizers.l1(l1_regu),
              bias_regularizer=regularizers.l1(l1_regu))(target)
    t = BatchNormalization(name='target-obs-bn1')(t)
    t = Activation('relu', name='target-obs-relu')(t)
    t = Dense(64, activation='relu', name='target-obs-dense2')(t)
    t = Dense(32, activation='relu', name='target-obs-dense3')(t)
    obs_all = Concatenate(name='obs-merge')([o, t])
    obs_all = Dense(100, activation='relu',
                    name='obs-latent-dense4',
                    kernel_regularizer=regularizers.l1(l1_regu),
                    bias_regularizer=regularizers.l1(l1_regu))(obs_all)
    obs_all = Dense(64, activation='relu', name='obs-latent-dense5')(obs_all)
    obs_all = Dense(32, activation='relu', name='obs-latent-dense6')(obs_all)

    obs_sub = Subtract(name='config-obs')([latent_config, obs_all])
    multi2 = Multiply(name='beta_sub')([beta, obs_sub])
    latent_action = Concatenate(name='theta_obs')([multi1, multi2])

    final = Dense(256, name='final-dense1',
                  kernel_regularizer=regularizers.l1(4*l1_regu))(latent_action)
    final = BatchNormalization(name='final-bn1')(final)
    final = Activation('relu', name='final-relu1')(final)
    final = Dense(128, activation='relu',
                  name='final-dense2',
                  kernel_regularizer=regularizers.l1(l1_regu),
                  bias_regularizer=regularizers.l1(l1_regu))(final)
    #final = Dropout(0.5, name='final-dp1')(final)
    final = Dense(64, activation='relu',
                  name='final-dense3',
                  kernel_regularizer=regularizers.l1(l1_regu),
                  bias_regularizer=regularizers.l1(l1_regu))(final)
    final = Dense(32, activation='relu', name='final-dense4')(final)
    final = Dense(dof, name='output')(final)

    model = Model(inputs=[config, target, obstacle],
                  outputs=final)
    """model.compile(loss=weighted_logcosh,
                  optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=lr_decay),
                  metrics=['mse'])"""
    return model


def model_with_latentspace2(dof):
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
    multi2 = Dense(32, name='final-dense1',
                   kernel_regularizer=regularizers.l1(l1_regu))(multi2)
    multi2 = BatchNormalization(name='final-bn1')(multi2)
    multi2 = Activation('relu', name='final-relu1')(multi2)
    multi2 = Dense(5, name='final-dense2')(multi2)
    final = Add(name='final-add')([multi1, multi2])
    """latent_action = Concatenate(name='theta_obs')([multi1, multi2])
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
    #final = Dense(32, activation='relu', name='final-dense4')(final)
    final = Dense(dof, name='output')(final)"""

    model = Model(inputs=[config, target, obstacle],
                  outputs=final)
    """model.compile(loss=weighted_logcosh,
                  optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=lr_decay),
                  metrics=['mse'])"""
    return model


def weighted_logcosh(y_true, y_pred):

    def _logcosh(x):
        return x + K.softplus(-2. * x) - K.log(2.)

    tfmetrics = tf.convert_to_tensor(metrics, dtype=tf.float32)
    sub = tf.subtract(y_pred, y_true)
    weighted_sub = 5*tf.multiply(tfmetrics, sub)
    return K.mean(_logcosh(weighted_sub), axis=-1)


def train_with_generator(datapath, batch_size, epochs):
    learning_rate = 2e-2  # 学习率
    lr_decay = 1e-3
    model = model_with_latentspace2(5)
    #model.load_weights('./h5files/5dof_latent_weights2.h5')
    model.compile(loss=weighted_logcosh,
                  optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=lr_decay),
                  metrics=['mse'])
    #model.summary()
    #print(learning_rate, lr_decay)
    #plot_model(model, to_file='5dof_latent3.jpg', show_shapes=True)
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
                                  callbacks=[TensorBoard(log_dir='./tensorboard_logs/5dof_latent8/log')],
                                  workers=3)
    # K.clear_session()
    #model.save('./h5files/5dof_latent_6.h5')
    model.save_weights('./h5files/5dof_latent_weights4.h5')


if __name__ == '__main__':
    datapath = '/home/czj/vrep_path_dataset/2/'
    train_with_generator(datapath, 100, 300)
