# %%
# Importing packages
# %%
from IASTrigCal import genIASTdata2D, PredLinIAST2D
import numpy as np
import os
import time
import matplotlib.pyplot as plt
# %%
# current path?
base_path = os.path.dirname(__file__)
#print(base_path)
#base_path = os.getcwd()
# print(os.path.dirname(__file__))
# %%
# Define Simple Isotherm Model
# %%
qm1 = 4
b1 = 0.2
qm2 = 1.5
b2 = 0.5

def iso1(P):
    numer = qm1*b1*P
    denom = 1+b1*P
    q_ret = numer/denom
    return q_ret

def iso2(P):
    numer = qm2*b2*P
    denom = 1+b2*P
    q_ret = numer/denom
    return q_ret

# %%
# y1 and P domain
# %%
y1_dom = np.linspace(0,1,51)
P_dom_low = np.linspace(0,1,26)
P_dom_high = np.linspace(1,30,30)
P_dom = np.concatenate([P_dom_low, P_dom_high])
# %%
# Generate IAST data
f_nam = 'IAST_da.pkl'

tic = time.time()
genIASTdata2D(iso1, iso2, y1_dom, P_dom, file_name=f_nam)
toc = time.time()-tic
print('CPU time = {0:.2f} min'.format(toc/60))

# %%
# Define a interpolation model
# %% 
q_model = PredLinIAST2D(file_name = f_nam)
# %%
P_test = np.linspace(0,10)
q1_test,q2_test = q_model.predict(0.1, 2)
q1_list = []
q2_list = []
for pp in P_test:
    q1_tmp,q2_tmp = q_model.predict(0.1, pp)
    q1_list.append(q1_tmp)
    q2_list.append(q2_tmp)
# %%
# Graph
