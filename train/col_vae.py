import pickle
import numpy as np
import keras
from processing.analyze_col import AnalyzeCol
from processing.angle_dis import obs_pt_big


def simple_ae(n_input):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(512, kernel_regularizer=keras.regularizers.l1(1e-3),name='dense1'))
    model.add(keras.layers.BatchNormalization(name='bn'))
    model.add(keras.layers.Dense(256, activation='relu',name='dense2'))