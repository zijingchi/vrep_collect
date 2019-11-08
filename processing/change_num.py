import os
import pickle
import numpy as np
import shutil
import re
"""
    REMEMBER TO CHECK IF THE DATAPATH IS WHAT YOU EXPECT!!!!!!!!!!
"""
datapath = '/home/czj/Downloads/ur5expert'
path2 = '/home/czj/Downloads/ur5expert'


def change_name(datapath):
    dirlist = os.listdir(datapath)
    dirlist.sort(key=lambda s: int(s))
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
                    '''a0 = data['actions'][0]
                    at = data['actions'][-1]
                    if np.linalg.norm(a0-at)<0.02 and np.random.rand()<0.8:
                        #shutil.rmtree(subdir)
                        continue'''
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
            train_list = train_list + list0['train'][:int(0.9*len(list0['train']))]
            vali_list = vali_list + list0['test'][:int(0.9*len(list0['test']))]
    with open(os.path.join(datapath, listpkl), 'wb') as f1:
        pickle.dump({'train': train_list, 'test': vali_list}, f1)


def separate_train_test3(datapath, listpkl, listpkl0=None):
    dirlist = os.listdir(datapath)
    id_list = []
    for d in dirlist:
        subdir = os.path.join(datapath, d)
        if os.path.isdir(subdir):
            datapkl = os.path.join(subdir, 'data.pkl')
            if os.path.exists(datapkl):
                id_list.append(d)
    id_size = len(id_list)
    train_size = int(0.8 * id_size)
    np.random.shuffle(id_list)
    train_list = id_list[:train_size]
    vali_list = id_list[train_size:]
    '''if listpkl0:
        with open(os.path.join(datapath, listpkl0), 'rb') as f0:
            list0 = pickle.load(f0)
            train_list = train_list + list0['train']
            vali_list = vali_list + list0['test']
    with open(os.path.join(datapath, listpkl), 'wb') as f1:
        pickle.dump({'train': train_list, 'test': vali_list}, f1)
    '''
    os.chdir(datapath)
    for d in train_list:
        shutil.copytree(d, os.path.join(path2, 'train', d))
    for d in vali_list:
        shutil.copytree(d, os.path.join(path2, 'val', d))


change_name(datapath)
#separate_train_test2(datapath, 'list2.pkl', 'list1.pkl')
