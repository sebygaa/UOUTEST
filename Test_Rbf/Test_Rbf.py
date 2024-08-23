# %%
from scipy.interpolate import Rbf
import os
import pickle
# %%
f = open('IAST_da.pkl', 'rb')
f_res = pickle.load(f)
f.close()
# %%
print(f_res['y1'].shape)
print(f_res['P'].shape)
print(f_res['q1'].shape)
q1 = f_res['q1']
q2 = f_res['q2']
P  = f_res['P']
y1 = f_res['y1']
P_list = []
y1_list = []
q1_list = []
q2_list = []

for ii, pp in enumerate(y1):
    for jj, yy in enumerate(P):
        q1_tmp = q1[ii,jj]
        q2_tmp = q2[ii,jj]
        P_list.append(pp)
        y1_list.append(yy)
        q1_list.append(q1_tmp)
        q2_list.append()
        
# %%
q1_rbf = Rbf(P,y1,q1)
#q2_rbf = Rbf()
