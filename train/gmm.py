from processing.analyze_col import AnalyzeCol
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


datapath = '/home/czj/col_check6'
ac = AnalyzeCol(datapath)
col_state1, i1 = ac.load_col_states(ac.files[5])
states = ac.load_pkl('states.pkl')
cs = states[col_state1]
n_mix = 10
gm = GaussianMixture(n_components=n_mix, tol=1e-4, max_iter=1000)
gm.fit(cs)
#p1=gm.score_samples(cs)

fig1 = plt.figure(1)
ax = Axes3D(fig1)
ax.scatter(cs[:, 0], cs[:, 1], cs[:, 2], s=5, alpha=0.5, c=np.log(gm.score_samples(cs)))
ax.set_xlabel('theta_1')
theta1_left, theta1_right = -1.6, 1.6
ax.set_xbound(theta1_left, theta1_right)
ax.set_ylabel('theta_2')
theta2_left, theta2_right = -1.9, 0.9
ax.set_ybound(theta2_left, theta2_right)
ax.set_zlabel('theta_3')
theta3_left, theta3_right = -2.7, 0.3
ax.set_zbound(theta3_left,theta3_right)

ax.scatter(gm.means_[:, 0], gm.means_[:, 1], gm.means_[:, 2], s=30, alpha=1., c=[0]*n_mix)
plt.show()

