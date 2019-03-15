import h5py
from keras.models import load_model
# import pickle
from train.rnntest import gru_test, weighted_logcosh

modelfile = '../train/h5files/gru_mid1.h5'


def files2h5(datapath, delete):
    pass


def deleteh5(basepath):
    pass


def openh5(h5path):
    hf = h5py.File(h5path, "r")
    a1 = hf["action/actions"]
    print(a1.attrs)


def main():
    model = load_model(modelfile)


if __name__ == '__main__':
    main()