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

learning_rate = 2e-2        # 学习率
l1_regu = 2e-5
lr_decay = 5e-3


def model_with_config_n_target(dof):
    config = Input(shape=(dof,), name='angles')
    target = Input(shape=(dof,), name='target')

    x1 = Dense(1024, name='config_dense1')(config)
    x1 = BatchNormalization(name='config_bn1')(x1)
    x1 = Activation(activation='relu', name='config_relu1')(x1)
    x1 = Dense(1024, kernel_regularizer=regularizers.l1(l1_regu), name='config_dense2')(x1)
    x1 = BatchNormalization(name='config_bn2')(x1)
    x1 = Activation(activation='relu', name='config_relu2')(x1)
    x1 = Dense(512, activation='relu', name='config_dense3')(x1)
    x1 = Dense(256, activation='relu', name='config_dense4')(x1)

    x4 = Dense(1024, name='sub_dense1')(target)
    x4 = BatchNormalization(name='sub_bn1')(x4)
    x4 = Activation('relu', name='sub_relu1')(x4)
    x4 = Dense(1024, kernel_regularizer=regularizers.l1(l1_regu), activation='relu', name='sub_dense2')(x4)
    x4 = BatchNormalization(name='sub_bn2')(x4)
    x4 = Activation(activation='relu', name='sub_relu2')(x4)
    x4 = Dense(512, activation='relu', name='sub_dense3')(x4)
    x4 = Dense(256, activation='relu', name='sub_dense4')(x4)

    y = Concatenate(name='merge')([x1, x4])
    y = Dense(256, name='final_dense1')(y)
    y = BatchNormalization(name='final_bn1')(y)
    y = Activation('relu', name='final_relu1')(y)
    y = Dense(128, activation='relu', name='final_dense2')(y)
    y = Dense(32, activation='relu', name='final_dense3')(y)

    final_output = Dense(dof, name='final_output')(y)
    model = Model(inputs=[config, target],
                  outputs=final_output)
    return model


def model_with_config_n_target2(dof):
    config = Input(shape=(dof,), name='angles')
    target = Input(shape=(dof,), name='target')
    obstacle_pos = Input(shape=(3,), name='obstacle_pos')
    obstacle_ori = Input(shape=(3,), name='obstacle_ori')
    x1 = Dense(512, name='config_dense1')(config)
    x1 = BatchNormalization(name='config_bn1')(x1)
    x1 = Activation(activation='relu', name='config_relu1')(x1)
    x1 = Dense(256, name='config_dense2')(x1)

    x1 = Activation(activation='relu', name='config_relu2')(x1)
    x1 = Dense(256, activation='relu',
               kernel_regularizer=regularizers.l1(l1_regu),
               name='config_dense3')(x1)
    x1 = Dense(256, name='config_dense4')(x1)
    x1 = BatchNormalization(name='config_bn2')(x1)
    x1 = Activation(activation='sigmoid', name='config_sigmoid1')(x1)

    x21 = Dense(512, name='obs_pos_dense1')(obstacle_pos)
    x21 = BatchNormalization(name='obs_pos_bn1')(x21)
    x21 = Activation(activation='relu', name='obs_pos_relu1')(x21)
    x21 = Dense(256, activation='relu', name='obs_pos_dense2')(x21)
    x21 = Dense(256, activation='relu', name='obs_pos_dense3')(x21)
    x22 = Dense(512, name='obs_ori_dense1')(obstacle_ori)
    x22 = BatchNormalization(name='obs_ori_bn1')(x22)
    x22 = Activation(activation='relu', name='obs_ori_relu1')(x22)
    x22 = Dense(256, activation='relu', name='obs_ori_dense2')(x22)
    x22 = Dense(256, activation='relu', name='obs_ori_dense3')(x22)
    x2 = Concatenate(name='obs')([x21, x22])
    x2 = Dense(1024,
               kernel_regularizer=regularizers.l1(l1_regu),
               name='obs_dense1')(x2)
    x2 = BatchNormalization(name='obs_bn1')(x2)
    x2 = Activation('relu', name='obs_relu1')(x2)
    x2 = Dense(512, activation='relu', name='obs_dense2')(x2)
    x2 = Dense(512, name='obs_dense3')(x2)
    x2 = BatchNormalization(name='obs_bn2')(x2)
    x2 = Activation('sigmoid', name='obs_sigmoid1')(x2)

    x3 = Dense(512,
               kernel_regularizer=regularizers.l1(l1_regu),
               name='tar_dense1')(target)
    x3 = BatchNormalization(name='tar_bn1')(x3)
    x3 = Activation('relu', name='tar_relu1')(x3)
    x3 = Dense(256, activation='relu', name='tar_dense2')(x3)
    x3 = Dense(256, activation='relu',
               kernel_regularizer=regularizers.l1(l1_regu),
               name='tar_dense3')(x3)
    x3 = Dense(128, name='tar_dense4')(x3)
    x3 = BatchNormalization(name='tar_bn2')(x3)
    x3 = Activation('sigmoid', name='tar_sigmoid1')(x3)

    merge1 = Concatenate(name='f1-merge1')([x1, x2, x3])
    act1 = Dense(1024, name='f1-dense1')(merge1)
    act1 = BatchNormalization(name='f1-bn1')(act1)
    act1 = Activation('relu', name='f1-relu1')(act1)
    act1 = Dense(512, activation='relu',
                 kernel_regularizer=regularizers.l1(l1_regu),
                 name='f1-dense2')(act1)
    act1 = Dense(256, activation='relu', name='f1-dense3')(act1)
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

    #final_output = Add(name='add')([act1, x4])
    final_output = Dense(dof, name='final_output')(act1)
    model = Model(inputs=[config, target, obstacle_pos, obstacle_ori],
                  outputs=final_output)
    """model.compile(loss=losses.logcosh,
                  optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=lr_decay),
                  metrics=['mae'])"""
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
    with open(os.path.join(datapath, 'list1.pkl'), 'rb') as f0:
        list0 = pickle.load(f0)
        train_list = train_list + list0['train']
        vali_list = vali_list + list0['test']
    with open(os.path.join(datapath, 'list2.pkl'), 'wb') as f1:
        pickle.dump({'train': train_list, 'test': vali_list}, f1)


def weighted_logcosh(y_true, y_pred):

    def _logcosh(x):
        return x + K.softplus(-2. * x) - K.log(2.)

    tfmetrics = tf.convert_to_tensor(metrics, dtype=tf.float32)
    sub = tf.subtract(y_pred, y_true)
    weighted_sub = 5*tf.multiply(tfmetrics, sub)
    return K.mean(_logcosh(weighted_sub), axis=-1)


def train_with_generator(datapath, batch_size, epochs, dof):
    model = model_with_config_n_target2(dof)
    #h5file = './h5files/5dof_model5.h5'
    #model.load_weights(h5file)
    model.compile(loss=losses.logcosh,
                  optimizer=optimizers.Adam(lr=2e-2, beta_1=0.9, beta_2=0.999, decay=5e-3),
                  metrics=['mse'])
    #plot_model(model, to_file='5dof_model1.jpg', show_shapes=True)
    with open(os.path.join(datapath, 'list1.pkl'), 'rb') as f:
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
    checkpoint = ModelCheckpoint(filepath='./h5files/3dof_mid_model.h5', monitor='val_mean_squared_error',
                                 save_best_only=True, save_weights_only=False, mode='min')
    model.fit_generator(generator=train_gen,
                        epochs=epochs,
                        validation_data=vali_gen,
                        use_multiprocessing=True,
                        callbacks=[TensorBoard(log_dir='./tensorboard_logs/3dof_model/log'), checkpoint],
                        workers=2)
    model.save('./h5files/3dof_model.h5')


if __name__ == '__main__':
    datapath = '/home/ubuntu/vdp/4_7/'

    train_with_generator(datapath, 100, 300, 3)
