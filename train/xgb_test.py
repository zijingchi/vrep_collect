import numpy as np
from processing.data_loader import PathPlanDset
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

expert_datapath = '/home/czj/Downloads/ur5exp3/'
expert_train_set = PathPlanDset(expert_datapath+'train')
expert_val_set = PathPlanDset(expert_datapath+'val')

model = MultiOutputRegressor(xgb.XGBRegressor(max_depth=20, learning_rate=0.01, n_estimators=200, silent=False, objective='reg:squarederror'))
#data_train = xgb.DMatrix(expert_set.train_set.inputs, expert_set.train_set.labels)
model.fit(expert_train_set.dset.inputs, expert_train_set.dset.labels)
#y_pred = model.predict(expert_set.val_set.inputs)
#err = y_pred - expert_set.val_set.labels
y_pred = model.predict(expert_val_set.dset.inputs)
err = y_pred - expert_val_set.dset.labels
sqe = np.sqrt(np.square(err))
print(sqe.mean(axis=0))
print(sqe.var(axis=0))

import time
sdx = expert_val_set.dset.inputs[100]
sdy = expert_val_set.dset.labels[100]
t1 = time.time()
pred_sdy = model.predict(sdx.reshape(1, -1))
t2 = time.time()
print(t2-t1)

'''
model = xgb.XGBRegressor(max_depth=20, learning_rate=0.01, n_estimators=200, silent=False,
                         objective='reg:squarederror')
model.fit(expert_set.train_set.inputs, expert_set.train_set.labels)
y_pred = model.predict(expert_set.val_set.inputs)
err = y_pred - expert_set.val_set.labels
sqe = np.square(err)
print(sqe.mean(axis=0))
'''