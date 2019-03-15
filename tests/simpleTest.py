
from processing.readDataFromFile import DataFromDirPkl
import numpy as np


datapath = '/home/ubuntu/vdp/4_7'

dataloader = DataFromDirPkl(datapath)
ni = dataloader.load('312')
configs = dataloader.configs_sequence(8, ni)
print(configs)