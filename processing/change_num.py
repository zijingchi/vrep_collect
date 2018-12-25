import os

datapath = '/home/czj/vrep_path_dataset/13/'
dirlist = os.listdir(datapath)
s = 276
numlist = list(range(s, s+len(dirlist)))

try:
    for d, n in zip(dirlist, numlist):
        oldname = os.path.join(datapath, d)
        newname = os.path.join(datapath, 'd'+str(n))
        os.rename(oldname, newname)
except OSError as e:
    print('error rename')
