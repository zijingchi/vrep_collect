import os
"""
    REMEMBER TO CHECK IF THE DATAPATH IS WHAT YOU EXPECT!!!!!!!!!!
"""
datapath = '/home/ubuntu/vrep_path_dataset/2/'
dirlist = os.listdir(datapath)
if 'list0.pkl' in dirlist:
    raise RuntimeError('not the new dataset!')
s = 0
numlist = list(range(s, s+len(dirlist)))

try:
    for d, n in zip(dirlist, numlist):
        oldname = os.path.join(datapath, d)
        newname = os.path.join(datapath, str(n))
        if os.path.isdir(oldname):
            os.rename(oldname, newname)
except OSError as e:
    print('error rename')
