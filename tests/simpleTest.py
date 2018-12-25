import pickle
import os

datapath = '/home/czj/vrep_path_dataset/11/'
dirlist = os.listdir(datapath)
os.chdir(datapath)
for d in dirlist:
    datapkl = d + '/data.pkl'
    daggerpkl = d + '/dagger.pkl'
    if os.path.exists(datapkl):
        with open(datapkl, 'rb') as f1, open(daggerpkl, 'wb') as f2:
            data = pickle.load(f1)
            configs = data['observations']
            actions = data['actions']
            for i in range(len(actions)):
                actions[i] = actions[i] - configs[i]
            data['actions'] = actions
            pickle.dump(data, f2)
