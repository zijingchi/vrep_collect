from processing.readDataFromFile import DataFromIndex
import numpy as np
import os
import shutil


def isDirect(d):
    if len(os.listdir(os.path.join(path, d, 'img1')))==0 or \
            not os.path.exists(os.path.join(path, d, 'data.pkl')):
        shutil.rmtree(os.path.join(path, d))
        return False
    ob1, ac1 = loader.read_per_index(d+'-2')
    if np.linalg.norm(ac1)>0.04:
        return False
    else:
        return True


path = '/home/czj/Downloads/ur5expert3'
loader = DataFromIndex(path)
directs = []
subdirs = os.listdir(path)
subdirs = [d for d in subdirs if os.path.isdir(os.path.join(path, d))]
for d in subdirs:
    if isDirect(d):
        directs.append(d)
print('{}/{}'.format(len(directs), len(subdirs)))
