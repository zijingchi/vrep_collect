# import matplotlib.pyplot as plt
import numpy as np
# import math
import keras
# import cv2
# import time
import os
# import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
# from keras import backend as K
from keras import optimizers, losses
from processing.readDataFromFile import DataFromDir
from processing.DataGenerator import CustomDataGenWthImg
# from keras.callbacks import LearningRateScheduler

img_width = 256
img_height = 256
learning_rate = 10**-np.random.randint(1, 7)*np.random.random(1)         # 学习率
# learning_rate = 0.1
lr_decay = np.random.random(1)


def vgg_modify_construct():
    #model = vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=None,
    #                    input_shape=(img_width, img_height, 3), pooling=None)

    model = Sequential()

    # BLOCK 1
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='block1_conv1',
                     input_shape=(img_width, img_height, 3)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool'))
    # 128x128

    # BLOCK2
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool'))
    # 64x64

    # BLOCK3
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool'))
    # 32x32

    # BLOCK4
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv2'))
    #model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool'))
    # 16x16

    # BLOCK5
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name='block5_conv2'))
    #model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool'))
    # 8x8

    model.add(Flatten())    # 8x8x128 -> 1x(8x8x128)
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc2'))
    #model.add(Dropout(0.5))
    #model.add(Dense(6, activation='softmax', name='vgg_out'))

    return model


def merge_model1():
    config = Input(shape=(6, ), name='angles')
    image1 = Input(shape=(img_width, img_height, 3), name='image1')
    image2 = Input(shape=(img_width, img_height, 3), name='image2')
    vgg_modified = vgg_modify_construct()
    out1 = vgg_modified(image1)
    out2 = vgg_modified(image2)
    image_concatenated = keras.layers.concatenate([out1, out2], axis=-1)
    merged1 = Dense(64, activation='relu')(image_concatenated)
    merged2 = Dense(16, activation='relu')(merged1)
    final_input = keras.layers.concatenate([merged2, config], axis=-1)
    x = Dense(32, activation='relu')(final_input)
    x = Dense(16, activation='relu')(x)
    final_output = Dense(6, name='final_output')(x)  # 激活函数需要实验
    final_model = Model(inputs=[config, image1, image2], outputs=final_output)
    final_model.compile(loss=losses.logcosh,
                        optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=lr_decay),
                        metrics=['mae'])
    return final_model


def merge_model2():
    configwithtarget = Input(shape=(12, ), name='angles')
    image1 = Input(shape=(img_width, img_height, 3), name='image1')
    image2 = Input(shape=(img_width, img_height, 3), name='image2')
    vgg_modified = vgg_modify_construct()
    out1 = vgg_modified(image1)
    out2 = vgg_modified(image2)
    image_concatenated = keras.layers.concatenate([out1, out2], axis=-1)
    merged1 = Dense(32, activation='relu')(image_concatenated)
    # merged2 = Dense(16, activation='relu')(merged1)
    final_input = keras.layers.concatenate([merged1, configwithtarget], axis=-1)
    x = Dense(32, activation='relu')(final_input)
    x = Dense(16, activation='relu')(x)
    final_output = Dense(6, activation='sigmoid', name='final_output')(x)
    final_model = Model(inputs=[configwithtarget, image1, image2], outputs=final_output)
    final_model.compile(loss=keras.losses.logcosh,
                        optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=lr_decay),
                        metrics=['mae'])
    return final_model


def load_data_from_dir(basepath):
    subdir = os.listdir(basepath)
    configs = []
    actions = []
    img1 = []
    img2 = []
    for dir in subdir:
        if (os.path.isdir(basepath + dir)) & (int(dir) < 200):
            data = DataFromDir(basepath, dir)
            data.load(True)
            configs.extend(data.configs)
            actions.extend(data.actions)
            img1.extend(data.img1)
            img2.extend(data.img2)
    configs = np.array(configs)
    actions = np.array(actions)
    img1 = np.array(img1)
    img2 = np.array(img2)
    return configs, img1, img2, actions


def train1(trainpath, valipath, batch_size, epochs):
    model = merge_model1()
    configs, img1, img2, actions = load_data_from_dir(trainpath)
    configs_v, img1_v, img2_v, actions_v = load_data_from_dir(valipath)
    validation_data = [[configs_v, img1_v, img2_v], actions_v]
    model.fit(x=[configs, img1, img2], y=actions, batch_size=batch_size, epochs=epochs, validation_data=validation_data)
    model.save('test1.h5')


def train_with_gen(datapath, batch_size, epochs):
    model = merge_model1()
    dirlist = os.listdir(datapath)
    id_list = []
    for d in dirlist:
        for i in range(50):
            id_list.append(d + '-' + str(i))
    id_size = len(id_list)
    train_size = int(0.7*id_size)
    np.random.shuffle(id_list)
    train_list = id_list[:train_size]
    vali_list = id_list[train_size:]
    train_gen = CustomDataGenWthImg(datapath=datapath,
                              list_IDs=train_list,
                              data_size=50,
                              dim=(256, 256, 3),
                              batch_size=batch_size)
    vali_gen = CustomDataGenWthImg(datapath=datapath,
                              list_IDs=vali_list,
                              data_size=50,
                              dim=(256, 256, 3),
                              batch_size=batch_size)
    model.fit_generator(generator=train_gen,
                        epochs=epochs,
                        validation_data=vali_gen,
                        use_multiprocessing=True,
                        workers=4)


def main():
    datapath = '/home/czj/vrep_path_dataset/4/'

    train_with_gen(datapath=datapath, batch_size=32, epochs=10)


if __name__ == '__main__':
    main()
