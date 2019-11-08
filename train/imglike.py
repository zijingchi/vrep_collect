import numpy as np
# import math
import keras
import os
import pickle
import tensorflow as tf
import keras.backend as K
from keras.models import Model, load_model
from keras.layers import BatchNormalization, Concatenate, Dense, Input, Activation, Flatten, Add, Multiply, Conv2D
from keras.callbacks import TensorBoard, ModelCheckpoint
from processing.fknodes import fktensor
from keras.utils import plot_model
from processing.angle_dis import metrics
from keras import optimizers, losses, regularizers
from processing.DataGenerator import GenImgLike
from train.training_imgless import weighted_logcosh
from train.mdn import *

l1_regu = 2e-4
N_MIXES = 10
OUTPUT_DIMS = 3


def construct(dof):
    concat_in = Input(shape=(5, 5, 1,), name='concat_in')
    x1 = Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l1(l1_regu), name='conv1')(concat_in)
    x1 = BatchNormalization(name='bn1')(x1)
    x1 = Activation(activation='tanh', name='tanh1')(x1)
    x1 = Conv2D(256, 3, padding='valid', kernel_regularizer=regularizers.l1(l1_regu), name='conv2')(x1)
    x1 = BatchNormalization(name='bn2')(x1)
    x1 = Activation('tanh', name='tanh2')(x1)
    x1 = Conv2D(512, 3, name='conv3')(x1)
    x1 = Flatten(name='flatten')(x1)
    x1 = BatchNormalization(name='bn3')(x1)
    x1 = Activation('tanh', name='tanh3')(x1)
    out = Dense(dof, name='output', kernel_regularizer=regularizers.l1(l1_regu))(x1)
    model = Model(concat_in, out)
    return model


def mdnconstruct():
    concat_in = Input(shape=(5, 5, 1,), name='concat_in')
    x1 = Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l1(l1_regu), name='conv1')(concat_in)
    x1 = BatchNormalization(name='bn1')(x1)
    x1 = Activation(activation='relu', name='relu1')(x1)
    x1 = Conv2D(256, 3, padding='valid', kernel_regularizer=regularizers.l1(l1_regu), name='conv2')(x1)
    x1 = BatchNormalization(name='bn2')(x1)
    x1 = Activation('relu', name='relu2')(x1)
    x1 = Conv2D(256, 3, name='conv3')(x1)
    x1 = Flatten(name='flatten')(x1)
    x1 = BatchNormalization(name='bn3')(x1)
    x1 = Activation('relu', name='relu3')(x1)
    act1_param = MDN(OUTPUT_DIMS, N_MIXES)(x1)
    model = Model(concat_in, act1_param)
    return model


def disconstruct():
    concat_in = Input(shape=(5, 5, 1,), name='concat_in')
    x1 = Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l1(l1_regu), name='conv1')(concat_in)
    x1 = BatchNormalization(name='bn1')(x1)
    x1 = Activation(activation='relu', name='relu1')(x1)
    x1 = Conv2D(256, 3, padding='valid', kernel_regularizer=regularizers.l1(l1_regu), name='conv2')(x1)
    x1 = BatchNormalization(name='bn2')(x1)
    x1 = Activation('relu', name='relu2')(x1)
    x1 = Conv2D(512, 3, name='conv3')(x1)
    x1 = Flatten(name='flatten')(x1)
    x1 = BatchNormalization(name='bn3')(x1)
    x1 = Activation('relu', name='relu3')(x1)
    y = Dense(5**3, kernel_regularizer=regularizers.l1(l1_regu), activation='softmax')(x1)
    model = Model(concat_in, y)
    return model


def tiperr(y_true, y_pred):
    co_true = fktensor(y_true)
    co_pred = fktensor(y_pred)
    return K.sqrt(K.sum(K.square(co_pred-co_true)))


def normerr(y_true, y_pred):
    uni_true = y_true/K.sqrt(K.batch_dot(y_true, y_true))
    uni_pred = y_pred/K.sqrt(K.batch_dot(y_pred, y_pred))
    return losses.mean_squared_error(uni_true, uni_pred)


def multicateloss(y_true, y_pred):
    sp_true = tf.split(y_true, 3, -1)
    sp_pred = tf.split(y_pred, 3, -1)
    return K.sum([K.categorical_crossentropy(t, p) for t, p in zip(sp_true, sp_pred)])


def train_with_generator(datapath, batch_size, epochs, dof, mdn=False):
    if mdn:
        model = mdnconstruct()
        #model.load_weights('./h5files/imglike_mdn_4mix_cpt.h5')
        model.compile(loss=get_mixture_mse_accuracy(OUTPUT_DIMS, N_MIXES),  # 'get_mixture_loss_func'
                      optimizer=optimizers.Adam(lr=2e-4, beta_1=0.9, beta_2=0.999, decay=8e-2),
                      metrics=[get_mixture_mse_accuracy(OUTPUT_DIMS, N_MIXES)])
    else:
        '''model = construct(dof)
        #model = load_model('./h5files/imglike_cpt0.h5')
        # h5file = './h5files/5dof_model5.h5'
        # model.load_weights(h5file)
        model.compile(loss=losses.logcosh,
                      optimizer=optimizers.Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, decay=4e-2),
                      metrics=['mse'])'''
        model = disconstruct()
        model.load_weights('./h5files/imglike_dis_cpt4.h5')
        model.compile(loss=losses.categorical_crossentropy,
                      optimizer=optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, decay=1e-2),
                      metrics=['accuracy']
                      )
    #plot_model(model, to_file='5dof_model1.jpg', show_shapes=True)
    with open(os.path.join(datapath, 'list1.pkl'), 'rb') as f:
        lists = pickle.load(f)
        train_list = lists['train']
        vali_list = lists['test']
    train_gen = GenImgLike(datapath=datapath,
                           list_IDs=train_list,
                           data_size=dof,
                           batch_size=batch_size)
    vali_gen = GenImgLike(datapath=datapath,
                          list_IDs=vali_list,
                          data_size=dof,
                          batch_size=batch_size)
    checkpoint = ModelCheckpoint(filepath='./h5files/imglike_dis_cpt4.h5', monitor='val_loss',
                                 save_best_only=True, save_weights_only=mdn, mode='min')
    model.fit_generator(generator=train_gen,
                        epochs=epochs,
                        validation_data=vali_gen,
                        use_multiprocessing=False,
                        callbacks=[TensorBoard(log_dir='./tensorboard_logs/imglike_dis_4/log'), checkpoint],
                        workers=2)
    model.save('./h5files/imglike_dis_4.h5')


if __name__ == '__main__':
    datapath = '/home/czj/Downloads/ur5expert3/'

    train_with_generator(datapath, 100, 20, 3, False)
