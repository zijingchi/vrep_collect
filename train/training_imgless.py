import numpy as np
# import math
import keras
import os
import pickle
import tensorflow as tf
import keras.backend as K
from keras.models import Model, load_model
from keras.layers import BatchNormalization, Concatenate, Dense, Input, Activation, Subtract, Add, Multiply
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import plot_model
from processing.angle_dis import metrics
from keras import optimizers, losses, regularizers
from processing.DataGenerator import CustomDataGenWthTarCfg
from train.mdn import *

l1_regu = 4e-5
N_MIXES = 8
OUTPUT_DIMS = 2

def model_with_config_n_target(dof):
    config = Input(shape=(dof,), name='angles')
    target = Input(shape=(dof,), name='target')

    x1 = Dense(1024, name='config_dense1')(config)
    x1 = BatchNormalization(name='config_bn1')(x1)
    x1 = Activation(activation='relu', name='config_relu1')(x1)
    x1 = Dense(1024,
               kernel_regularizer=regularizers.l1(l1_regu),
               bias_regularizer=regularizers.l1(l1_regu),
               name='config_dense2')(x1)
    x1 = BatchNormalization(name='config_bn2')(x1)
    x1 = Activation(activation='relu', name='config_relu2')(x1)
    x1 = Dense(512, activation='relu', name='config_dense3')(x1)
    x1 = Dense(256, activation='relu', name='config_dense4')(x1)

    x4 = Dense(1024, name='sub_dense1')(target)
    x4 = BatchNormalization(name='sub_bn1')(x4)
    x4 = Activation('relu', name='sub_relu1')(x4)
    x4 = Dense(1024,
               kernel_regularizer=regularizers.l1_l2(l1_regu),
               bias_regularizer=regularizers.l2(l1_regu),
               activation='relu', name='sub_dense2')(x4)
    x4 = BatchNormalization(name='sub_bn2')(x4)
    x4 = Activation(activation='relu', name='sub_relu2')(x4)
    x4 = Dense(512, activation='relu', name='sub_dense3')(x4)
    x4 = Dense(256, activation='relu', name='sub_dense4')(x4)

    y = Concatenate(name='merge')([x1, x4])
    y = Dense(256,
              kernel_regularizer=regularizers.l2(l1_regu),
              bias_regularizer=regularizers.l2(l1_regu),
              name='final_dense1')(y)
    y = BatchNormalization(name='final_bn1')(y)
    y = Activation('relu', name='final_relu1')(y)
    y = Dense(128, activation='relu', name='final_dense2')(y)
    y = Dense(32, activation='relu', name='final_dense3')(y)

    final_output = Dense(dof, name='final_output')(y)
    model = Model(inputs=[config, target],
                  outputs=final_output)
    return model


def simple_model(dof, input_size, model_type='straight'):
    concat_in = Input(shape=(input_size,), name='concat_in')
    x1 = Dense(512, name='dense1', kernel_regularizer=regularizers.l1(l1_regu))(concat_in)
    x1 = BatchNormalization(name='bn1')(x1)
    x1 = Activation(activation='tanh', name='tanh1')(x1)
    x1 = Dense(256, kernel_regularizer=regularizers.l1(l1_regu), name='dense2')(x1)
    x1 = BatchNormalization(name='bn2')(x1)
    x1 = Activation('tanh', name='tanh2')(x1)
    x1 = Dense(256, kernel_regularizer=regularizers.l1(l1_regu), activation='relu', name='dense3')(x1)
    x1 = BatchNormalization(name='bn3')(x1)
    x1 = Activation('tanh', name='tanh3')(x1)
    if model_type=='mdn':
        act1_param = MDN(OUTPUT_DIMS, N_MIXES)(x1)
        model = Model(concat_in, act1_param)
    else:
        out = Dense(dof, name='output')(x1)
        model = Model(concat_in, out)

    return model


def simple_model_discrete(dof, input_size):
    concat_in = Input(shape=(input_size,), name='concat_in')
    x1 = Dense(512, name='dense1', kernel_regularizer=regularizers.l2(l1_regu))(concat_in)
    x1 = BatchNormalization(name='bn1')(x1)
    x1 = Activation(activation='relu', name='relu1')(x1)
    x1 = Dense(512, kernel_regularizer=regularizers.l2(l1_regu), name='dense2')(x1)
    x1 = BatchNormalization(name='bn2')(x1)
    x1 = Activation('relu', name='relu2')(x1)
    x1 = Dense(256, kernel_regularizer=regularizers.l2(l1_regu), activation='relu', name='dense3')(x1)
    x1 = BatchNormalization(name='bn3')(x1)
    x1 = Activation('relu', name='relu3')(x1)
    '''y1 = Dense(7, activation='softmax', name='axis1')(x1)
    y2 = Dense(7, activation='softmax', name='axis2')(x1)
    y3 = Dense(7, activation='softmax', name='axis3')(x1)
    y = Concatenate(name='concat')([y1, y2, y3])'''
    y = Dense(5*dof, name='output')(x1)
    y = Activation('sigmoid')(y)
    model = Model(concat_in, y)
    return model


def multicateloss(y_true, y_pred):
    sp_true = tf.split(y_true, 3, -1)
    sp_pred = tf.split(y_pred, 3, -1)
    return K.max([K.categorical_crossentropy(t, p) for t, p in zip(sp_true, sp_pred)])


def categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())


def normerr(y_true, y_pred):
    uni_true = y_true/K.sqrt(K.batch_dot(y_true, y_true, axes=1))
    uni_pred = y_pred/K.sqrt(K.batch_dot(y_pred, y_pred, axes=1))
    return weighted_logcosh(uni_true, uni_pred)


def multiacc(y_true, y_pred):
    sp_true = tf.split(y_true, 3, -1)
    sp_pred = tf.split(y_pred, 3, -1)
    return categorical_accuracy(sp_true[0], sp_pred[0])*categorical_accuracy(sp_true[1], sp_pred[1])*categorical_accuracy(sp_true[2], sp_pred[2])


def model_with_config_n_target2(dof):
    config = Input(shape=(dof,), name='angles')
    target = Input(shape=(dof,), name='target')
    obstacle_pos = Input(shape=(3,), name='obstacle_pos')
    obstacle_ori = Input(shape=(3,), name='obstacle_ori')
    x1 = Dense(256, name='config_dense1')(config)
    x1 = BatchNormalization(name='config_bn1')(x1)
    x1 = Activation(activation='tanh', name='config_relu1')(x1)
    x1 = Dense(128, kernel_regularizer=regularizers.l2(l1_regu),
               bias_regularizer=regularizers.l2(l1_regu),
               name='config_dense2')(x1)

    x1 = Activation(activation='relu', name='config_relu2')(x1)
    x1 = Dense(64, activation='relu',
               kernel_regularizer=regularizers.l2(l1_regu),
               bias_regularizer=regularizers.l2(l1_regu),
               name='config_dense3')(x1)
    x1 = Dense(64, name='config_dense4')(x1)
    x1 = BatchNormalization(name='config_bn2')(x1)
    x1 = Activation(activation='sigmoid', name='config_sigmoid1')(x1)

    x21 = Dense(256, name='obs_pos_dense1')(obstacle_pos)
    x21 = BatchNormalization(name='obs_pos_bn1')(x21)
    x21 = Activation(activation='tanh', name='obs_pos_relu1')(x21)
    x21 = Dense(128, activation='relu',
                kernel_regularizer=regularizers.l2(l1_regu),
                bias_regularizer=regularizers.l2(l1_regu),
                name='obs_pos_dense2')(x21)
    x21 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l1_regu),
                bias_regularizer=regularizers.l2(l1_regu),
                name='obs_pos_dense3')(x21)
    x22 = Dense(256, name='obs_ori_dense1')(obstacle_ori)
    x22 = BatchNormalization(name='obs_ori_bn1')(x22)
    x22 = Activation(activation='tanh', name='obs_ori_relu1')(x22)
    x22 = Dense(128, kernel_regularizer=regularizers.l2(l1_regu),
                bias_regularizer=regularizers.l2(l1_regu),
                activation='relu', name='obs_ori_dense2')(x22)
    x22 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l1_regu),
                bias_regularizer=regularizers.l2(l1_regu),
                name='obs_ori_dense3')(x22)
    x2 = Concatenate(name='obs')([x21, x22])
    x2 = Dense(512,
               kernel_regularizer=regularizers.l1(l1_regu),
               name='obs_dense1')(x2)
    x2 = BatchNormalization(name='obs_bn1')(x2)
    x2 = Activation('tanh', name='obs_relu1')(x2)
    x2 = Dense(256, activation='relu', name='obs_dense2')(x2)
    x2 = Dense(64, name='obs_dense3')(x2)
    x2 = BatchNormalization(name='obs_bn2')(x2)
    x2 = Activation('sigmoid', name='obs_sigmoid1')(x2)

    x3 = Dense(256,
               kernel_regularizer=regularizers.l1(l1_regu),
               name='tar_dense1')(target)
    x3 = BatchNormalization(name='tar_bn1')(x3)
    x3 = Activation('tanh', name='tar_relu1')(x3)
    x3 = Dense(256, activation='relu', name='tar_dense2')(x3)
    x3 = Dense(128, activation='relu',
               kernel_regularizer=regularizers.l1(l1_regu),
               name='tar_dense3')(x3)
    x3 = Dense(64, name='tar_dense4')(x3)
    x3 = BatchNormalization(name='tar_bn2')(x3)
    x3 = Activation('sigmoid', name='tar_sigmoid1')(x3)

    merge1 = Concatenate(name='f1-merge1')([x1, x2, x3])
    act1 = Dense(256,
                 kernel_regularizer=regularizers.l2(l1_regu),
                 bias_regularizer=regularizers.l2(l1_regu),
                 name='f1-dense1')(merge1)
    act1 = BatchNormalization(name='f1-bn1')(act1)
    act1 = Activation('relu', name='f1-relu1')(act1)
    act1 = Dense(128, activation='relu',
                 kernel_regularizer=regularizers.l2(l1_regu),
                 bias_regularizer=regularizers.l2(l1_regu),
                 name='f1-dense2')(act1)
    act1 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l1_regu),
                 bias_regularizer=regularizers.l2(l1_regu),
                 name='f1-dense3')(act1)
    #act1 = BatchNormalization(name='f1-bn1')(act1)

    """sub = Subtract(name='sub')([target, config])
    x4 = Dense(64,
               kernel_regularizer=regularizers.l1(l1_regu),
               name='sub_dense1')(merge1)
    x4 = BatchNormalization(name='sub_bn1')(x4)
    x4 = Activation('relu', name='sub_relu1')(x4)
    x4 = Dense(32, activation='relu', name='sub_dense2')(x4)
    beta = Dense(1, kernel_regularizer=regularizers.l1(l1_regu),
                 bias_regularizer=regularizers.l1(l1_regu),
                 activation='relu', name='sub_dense3')(x4)"""

    act1 = Dense(dof, name='final_output')(act1)
    model = Model(inputs=[config, target, obstacle_pos, obstacle_ori],
                  outputs=act1)
    """model.compile(loss=losses.logcosh,
                  optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=lr_decay),
                  metrics=['mae'])"""
    return model


def weighted_logcosh(y_true, y_pred):

    def _logcosh(x):
        return x + K.softplus(-2. * x) - K.log(2.)

    tfmetrics = tf.convert_to_tensor(metrics, dtype=tf.float32)
    sub = tf.subtract(y_pred, y_true)
    weighted_sub = 5*tf.multiply(tfmetrics, sub)
    return K.mean(_logcosh(weighted_sub), axis=-1)


def train_with_generator(datapath, batch_size, epochs, dof, model_type='straight'):
    # model = model_with_config_n_target2(dof)
    if model_type=='straight':
        model = simple_model(dof, 25)
        #model = load_model('./h5files/3dof_model10.h5')
        h5file = './h5files/5dof_simple_norm_cpt3.h5'
        model.load_weights(h5file)
        model.compile(loss=normerr,
                      optimizer=optimizers.Adam(lr=4e-4, beta_1=0.9, beta_2=0.999, decay=8e-2),
                      metrics=[normerr])
        #plot_model(model, to_file='5dof_model1.jpg', show_shapes=True)
    elif model_type=='mdn':
        model = simple_model(dof, 25, 'mdn')
        model.compile(loss=get_mixture_mse_accuracy(OUTPUT_DIMS, N_MIXES),
                      optimizer=optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, decay=1e-2),
                      metrics=[get_mixture_mse_accuracy(OUTPUT_DIMS, N_MIXES)])
    else:
        model = simple_model_discrete(dof, 25)
        model.compile(loss=losses.binary_crossentropy,
                      optimizer=optimizers.Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, decay=1e-2),
                      metrics=[multiacc])
    with open(os.path.join(datapath, 'list2.pkl'), 'rb') as f:
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
    checkpoint = ModelCheckpoint(filepath='./h5files/5dof_simple_norm_cpt3.h5', monitor='val_normerr',
                                 save_best_only=True, save_weights_only=True, mode='min')
    model.fit_generator(generator=train_gen,
                        epochs=epochs,
                        validation_data=vali_gen,
                        use_multiprocessing=False,
                        callbacks=[TensorBoard(log_dir='./tensorboard_logs/5dof_norm_simple3/log'), checkpoint],
                        workers=2)
    model.save_weights('./h5files/5dof_simple_norm2.h5')


def train_single_dimension(datapath, batch_size, epochs):
    from processing.data_loader import PathPlanDset
    dataset = PathPlanDset(datapath)
    model = simple_model(1)
    model.compile(loss=losses.logcosh,
                  optimizer=optimizers.Adam(lr=2e-3, beta_1=0.9, beta_2=0.999, decay=1e-2),
                  metrics=['mse'])
    model.fit(dataset.train_set.inputs, dataset.val_set.labels, batch_size, epochs)


if __name__ == '__main__':
    datapath = '/home/czj/Downloads/ur5expert3/'

    train_with_generator(datapath, 100, 50, 5, model_type='straight')
