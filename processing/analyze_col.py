import pickle
import numpy as np
import os
import re


class AnalyzeCol(object):

    def __init__(self, upperdir):
        self.upperdir = upperdir
        allfiles = os.listdir(upperdir)
        col_files = []
        obs_files = None
        states_files = None
        for file in allfiles:
            if 'col_states' in file:
                col_files.append(file)
            elif file == 'states.pkl':
                states_files = file
            elif file == 'obs.pkl':
                obs_files = file

        if obs_files and states_files:
            raise FileExistsError('file not exist')

        self.files = col_files
        os.chdir(upperdir)

    def load_pkl(self, fname):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        return data

    def load_col_states(self, fname):
        match = re.compile('\d*')
        order = int(match.findall(fname)[0])
        data = self.load_pkl(fname)
        col_states_i = np.nonzero(data)[0]
        return col_states_i, order

    def organize_col(self):
        states = self.load_pkl('states.pkl')
        obs = self.load_pkl('obs.pkl')
        all_col_states = []
        for fname in self.files:
            col_states_i, i = self.load_col_states(fname)
            all_col_states.append([states[col_states_i], obs[i]])

        self.all_col_states = all_col_states

