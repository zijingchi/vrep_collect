import os
import pickle
import numpy as np
"""
    REMEMBER TO CHECK IF THE DATAPATH IS WHAT YOU EXPECT!!!!!!!!!!
"""
datapath = '/home/ubuntu/vdp/3'


def change_name(datapath):
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


def separate_train_test2(datapath, listpkl, listpkl0=None):
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
    #np.random.shuffle(id_list)
    train_list = id_list[:train_size]
    vali_list = id_list[train_size:]
    if listpkl0:
        with open(os.path.join(datapath, listpkl0), 'rb') as f0:
            list0 = pickle.load(f0)
            train_list = train_list + list0['train']
            vali_list = vali_list + list0['test']
    with open(os.path.join(datapath, listpkl), 'wb') as f1:
        pickle.dump({'train': train_list, 'test': vali_list}, f1)


change_name(datapath)
separate_train_test2(datapath, 'list0.pkl')
