import os
import pickle
import tensorflow as tf
import keras.backend as K
from keras.models import Model, load_model
from keras.layers import BatchNormalization, Concatenate, Dense, Input, Lambda, Subtract, Add, Dot, LeakyReLU
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import optimizers, losses, regularizers
from processing.DataGenerator import CustomDataGen4Pol

regu_coef = 1e-5


def construct_potential():
    configin = Input(shape=(5, ), name='config')
    goalin = Input(shape=(5, ), name='goal')
    obsin = Input(shape=(3, ), name='obs_pos')
    fkps = Input(shape=(12, ), name='fk')

    config = Dense(128, kernel_regularizer=regularizers.l2(regu_coef), name='config-dense1')(configin)
    config = BatchNormalization(name='config-bn1')(config)
    config = LeakyReLU(0.2)(config)
    config = Dense(128, kernel_regularizer=regularizers.l2(regu_coef), name='config-dense2')(config)
    config = BatchNormalization(name='config-bn2')(config)
    fk = Dense(128, kernel_regularizer=regularizers.l2(regu_coef), name='fk-dense1')(fkps)
    fk = BatchNormalization(name='fk-bn1')(fk)
    fk = LeakyReLU(0.2)(fk)
    fk = Dense(128, kernel_regularizer=regularizers.l2(regu_coef), name='fk-dense2')(fk)
    fk = BatchNormalization(name='fk-bn2')(fk)
    obs = Dense(256, kernel_regularizer=regularizers.l2(regu_coef), name='obs-dense1')(obsin)
    obs = BatchNormalization(name='obs-bn1')(obs)
    obs = Dense(256, kernel_regularizer=regularizers.l2(regu_coef), name='obs-dense2')(obs)
    obs = BatchNormalization(name='obs-bn2')(obs)
    obs = LeakyReLU(0.2)(obs)
    concat = Concatenate(name='concat')([obs, config, fk])
    concat = LeakyReLU(0.2)(concat)
    y = Dense(512, kernel_regularizer=regularizers.l2(regu_coef), name='y1')(concat)
    y = BatchNormalization(name='y-bn1')(y)
    y = LeakyReLU(0.2)(y)
    y = Dense(256, kernel_regularizer=regularizers.l2(regu_coef), name='y-dense2')(y)
    y = BatchNormalization(name='y-bn2')(y)
    y = LeakyReLU(0.2)(y)
    y = Dense(5, kernel_regularizer=regularizers.l2(regu_coef), name='output')(y)
    sub = Subtract()([goalin, configin])
    y = Add()([sub, y])

    model = Model(inputs=[configin, goalin, obsin, fkps], outputs=y)
    return model


def normerr(y_true, y_pred):
    uni_true = K.l2_normalize(y_true, axis=1)
    uni_pred = K.l2_normalize(y_pred, axis=1)
    return tf.arccos(K.batch_dot(uni_true, uni_pred, axes=-1))
    #return losses.mean_squared_error(uni_true, uni_pred)


def train_with_generator(datapath, batch_size, epochs):
    model = construct_potential()
    model.load_weights('./h5files/pol_cpt0.h5')
    model.compile(loss=normerr,
                  optimizer=optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, decay=8e-2),
                  metrics=[normerr])
    with open(os.path.join(datapath, 'list0.pkl'), 'rb') as f:
        lists = pickle.load(f)
        train_list = lists['train']
        vali_list = lists['test']
    train_gen = CustomDataGen4Pol(datapath=datapath,
                                  list_IDs=train_list,
                                  batch_size=batch_size)
    vali_gen = CustomDataGen4Pol(datapath=datapath,
                                 list_IDs=vali_list,
                                 batch_size=batch_size)
    checkpoint = ModelCheckpoint(filepath='./h5files/pol_cpt1.h5', monitor='val_normerr',
                                 save_best_only=True, save_weights_only=True, mode='min')
    model.fit_generator(generator=train_gen,
                        epochs=epochs,
                        validation_data=vali_gen,
                        use_multiprocessing=False,
                        callbacks=[TensorBoard(log_dir='./tensorboard_logs/pol1/log'), checkpoint],
                        workers=2)
    model.save_weights('./h5files/pol1.h5')


if __name__ == '__main__':
    datapath = '/home/czj/Downloads/ur5expert/'

    train_with_generator(datapath, 100, 50)