import keras
import os
from train.train3 import weighted_logcosh

modelh5 = '../train/h5files/5dof_latent_2.h5'
if os.path.exists(modelh5):
    model = keras.models.load_model(modelh5)
    print('load')