import numpy as np
import os
import pickle
import tensorflow as tf
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import BatchNormalization, Concatenate, Dense, Input, Activation, TimeDistributed, GRU, Masking, LSTM, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import plot_model
from keras import optimizers, losses, regularizers
from processing.angle_dis import metrics
from processing.DataGenerator import CustomDataGenWthTarCfgSqc


def config_process(name):
    model = Sequential()
    with tf.name_scope(name):
        model.add(Dense(512, name='config_dense1'))
        model.add(BatchNormalization(name='config_bn1'))
        model.add(Activation('relu', name='config_relu1'))
        model.add(Dense(256, activation='relu', name='config_dense2'))
        model.add(Dense(128, activation='relu', name='config_dense3'))
    return model


def static_state(dof, l1_regu):
    target = Input(shape=(dof,), name='target')
    obstacle_pos = Input(shape=(3,), name='obstacle_pos')
    obstacle_ori = Input(shape=(3,), name='obstacle_ori')

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
    x2 = Dense(256,
               kernel_regularizer=regularizers.l1(l1_regu),
               name='obs_dense1')(x2)
    x2 = BatchNormalization(name='obs_bn1')(x2)
    x2 = Activation('relu', name='obs_relu1')(x2)
    x2 = Dense(256, activation='relu', name='obs_dense2')(x2)
    x2 = Dense(128, activation='relu', name='obs_dense3')(x2)

    target_process = config_process('target')
    target_h = target_process(target)
    merge1 = Concatenate(name='h0_merge')([x2, target_h])
    h0 = Dense(256, kernel_regularizer=regularizers.l1(l1_regu), name='h0_dense1')(merge1)
    h0 = BatchNormalization(name='h0_bn1')(h0)
    h0 = Activation('relu', name='h0_relu1')(h0)
    h0 = Dense(256, activation='relu', name='h0_dense2')(h0)

    model = Model(inputs=[target, obstacle_pos, obstacle_ori],
                  outputs=h0)
    return model


def gru_test(dof, maxstep, l1_regu):
    configs = Input(shape=(maxstep, dof,), name='configs')
    target = Input(shape=(dof,), name='target')
    obstacle_pos = Input(shape=(3,), name='obstacle_pos')
    obstacle_ori = Input(shape=(3,), name='obstacle_ori')

    h0 = static_state(dof, l1_regu)([target, obstacle_pos, obstacle_ori])
    configs_m = Masking(mask_value=0., name='masking')(configs)
    configs_h = TimeDistributed(config_process('configs'))(configs_m)
    out = GRU(512, name='gru')(configs_h, initial_state=h0)
    out = Dense(128, name='out_dense1')(out)
    out = Dense(dof, name='output')(out)

    model = Model(inputs=[configs, target, obstacle_pos, obstacle_ori],
                  outputs=out)
    return model


def lstm_test(dof, maxstep, l1_regu):
    configs = Input(shape=(maxstep, dof,), name='configs')
    target = Input(shape=(dof,), name='target')
    obstacle_pos = Input(shape=(3,), name='obstacle_pos')
    obstacle_ori = Input(shape=(3,), name='obstacle_ori')

    h0 = static_state(dof, l1_regu)([target, obstacle_pos, obstacle_ori])
    configs_m = Masking(mask_value=0., name='masking')(configs)
    configs_h = TimeDistributed(config_process('configs'))(configs_m)
    out = LSTM(128, name='lstm1')(configs_h, initial_state=[h0[:, :128], h0[:, 128:]])
    #out = LSTM(128, name='lstm2')(out)
    out = Dense(64, name='out_dense1')(out)
    out = Dropout(0.5, name='dropout')(out)
    out = Dense(3, name='output')(out)

    model = Model(inputs=[configs, target, obstacle_pos, obstacle_ori],
                  outputs=out)
    return model


def weighted_logcosh(y_true, y_pred):

    def _logcosh(x):
        return x + K.softplus(-2. * x) - K.log(2.)

    tfmetrics = tf.convert_to_tensor(metrics, dtype=tf.float32)
    sub = tf.subtract(y_pred, y_true)
    weighted_sub = 8*tf.multiply(tfmetrics, sub)
    return K.mean(_logcosh(weighted_sub), axis=-1)


def train_with_generator(datapath, batch_size, epochs, maxstep, dof):
    l1_regu = 1e-5
    model = lstm_test(dof, maxstep, l1_regu)
    model.compile(loss=losses.logcosh,
                  optimizer=optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, decay=1e-3),
                  metrics=['mse'])
    #model.load_weights('./h5files/gru_weight_mid3.h5')
    #plot_model(model, to_file='gru_model1.jpg')
    with open(os.path.join(datapath, 'seqlist1.pkl'), 'rb') as f:
        lists = pickle.load(f)
        train_list = lists['train']
        vali_list = lists['test']
    train_gen = CustomDataGenWthTarCfgSqc(datapath=datapath,
                                          list_IDs=train_list,
                                          dof=dof,
                                          max_step=maxstep,
                                          batch_size=batch_size)
    vali_gen = CustomDataGenWthTarCfgSqc(datapath=datapath,
                                         list_IDs=vali_list,
                                         dof=dof,
                                         max_step=maxstep,
                                         batch_size=batch_size)
    checkpoint = ModelCheckpoint(filepath='./h5files/lstm_weight_mid2.h5', monitor='val_mean_squared_error',
                                 save_best_only=True, save_weights_only=True, mode='min')
    model.fit_generator(generator=train_gen,
                        epochs=epochs,
                        validation_data=vali_gen,
                        use_multiprocessing=True,
                        callbacks=[TensorBoard(log_dir='./tensorboard_logs/lstm2/log'), checkpoint],
                        workers=3)
    model.save('./h5files/lstm_model2.h5')


if __name__ == '__main__':
    datapath = '/home/ubuntu/vdp/4_7'
    train_with_generator(datapath, 10, 200, 8, 5)
