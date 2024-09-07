# %%
# Importing packages
# %%
from IASTrigCal import genIASTdata3D, PredLinIAST3D, IAST_bi
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
qm1 = 3
b1 = 0.1
qm2 = 1.5
b2 = 0.2

dH1 = 15.0e3
dH2 = 4.0e3
R_gas = 8.3145 
T_ref = 300
def iso1(P, T):
    P_norm = P*np.exp(dH1/R_gas*(1/T-1/T_ref))
    numer = qm1*b1*P_norm
    denom = 1+b1*P_norm
    q_ret = numer/denom
    return q_ret

def iso2(P, T):
    P_norm = P*np.exp(dH2/R_gas*(1/T-1/T_ref))
    numer = qm2*b2*P_norm
    denom = 1+b2*P_norm
    q_ret = numer/denom
    return q_ret
# %%
# TEST low concentration
# %%

iso1_T_fix = lambda P: iso1(P,280)
iso2_T_fix = lambda P: iso2(P,280)
q_res_test = IAST_bi(iso1_T_fix, iso2_T_fix, 0.01, 2)

# %%
# y1 and P domain
# %%
y1_dom = np.linspace(0,1,101)
P_dom_low = np.linspace(0,1,26)
P_dom_high = np.linspace(1,30,60)
P_dom = np.concatenate([P_dom_low, P_dom_high[1:]])
T_dom = np.linspace(283,383, 21)
# %%
# File Name
# %%
f_nam = 'IAST_da_3D.pkl'
# %%
# Generate IAST data
'''
tic = time.time()
genIASTdata3D(iso1, iso2, y1_dom, P_dom,T_dom, file_name=f_nam)
toc = time.time()-tic
print('CPU time = {0:.2f} min'.format(toc/60))
'''

# %%
# Define a interpolation model
# %% 
q_model = PredLinIAST3D(file_name = f_nam)
# %%
P_test = np.linspace(0,10,51)
T_test = [280, 300, 320, 340]
q1_test,q2_test = q_model.predict(0.1, 2, 300)
q1_list = []
q2_list = []

for TT in T_test:
    q1_list_tmp = []
    q2_list_tmp = []
    for pp in P_test:
        q1_tmp,q2_tmp = q_model.predict(0.1, pp, TT)
        q1_list_tmp.append(q1_tmp)
        q2_list_tmp.append(q2_tmp)
    q1_list.append(q1_list_tmp)
    q2_list.append(q2_list_tmp)

# %%
# Visualization
plt.figure()
for ii, TT in enumerate(T_test):
    plt.plot(P_test, q1_list[ii], linewidth = 2,
            label ='q1 at T = {0:.0f}'.format(TT))
plt.legend(fontsize = 13)
plt.savefig('Test_isoLin3D_q1.png', dpi=150, bbox_inches='tight')

plt.figure()
for ii, TT in enumerate(T_test):
    plt.plot(P_test, q2_list[ii], linewidth = 2,
            label='q2 at T = {0:.0f}'.format(TT))
plt.legend(fontsize = 13)
plt.savefig('Test_isoLin3D_q2.png', dpi=150, bbox_inches='tight')

# %%
