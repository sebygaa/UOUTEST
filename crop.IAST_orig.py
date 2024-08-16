# %%
# Importing
# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import pickle
import seaborn as sns
# %%
# %%
# File path
# %%
f_path = 'res_IAST_orig'
f_path = 'res_oriIAST24081616'
#base_path = os.getcwd()
base_path = os.path.dirname(__file__)

# %%
# go to the directory and read all data
# And Classify them
# %%
os.chdir(base_path)
os.chdir(f_path)
fnam_list = os.listdir()

pkl_list = []
txt_list = []
other_list =[]
for fff in fnam_list:
    if fff[-4:] == '.txt':
        txt_list.append(fff)
    elif fff[-4:] == '.pkl':
        pkl_list.append(fff)
    else:
        other_list.append(fff)
print(other_list)
os.chdir(base_path)

# %%
# Function for text to input variable
# %%
txt_tmp = txt_list[0]
print(txt_tmp)

def txt2vk(txt_tmp):
    i_v_start = -1
    for ii, tt in enumerate(txt_tmp):
        if tt == 'v':
            i_v_start = ii + 1
        elif tt == 'k':
            i_v_end =  ii -1
            i_k_start = ii +1
        elif tt == '.':
            i_k_end = ii
    if i_v_start < -0.001:
        return False, False
    v_re = float(txt_tmp[i_v_start:i_v_end])
    k_re = float(txt_tmp[i_k_start:i_k_end])
    return v_re, k_re
    
v_test,k_test = txt2vk(txt_tmp)
print(v_test)
print(k_test)
# %%
# List up all inputs
v_set = set([])
k_set = set([])
input_list= []

for txtt in txt_list:
    v_tmp, k_tmp = txt2vk(txtt)
    if type(v_tmp) == type(False):
        continue
    v_set.add(v_tmp)
    k_set.add(k_tmp)
    input_list.append([v_tmp, k_tmp])
v_list = list(v_set)
v_list.sort()
k_list = list(k_set)
k_list.sort()

print(v_list)
print(k_list)

di_CPUmin = {}
di_CPUmin['velocity (m/s)'] = v_list
 
#print(input_list)

# %%
# Crop the data
# %%
os.chdir(base_path)
os.chdir(f_path)

CPU_minu_list = []
ind_list = []

CPU_minu_mat = np.zeros([len(v_list),len(k_list)])

for ff,inpp in zip(txt_list, input_list):
    r = open(ff,'r')
    line1 = r.readline()
    line2 = r.readline()
    #print(line1)
    #print(line2)
    ii_end = 0
    for ii,ll in enumerate(line2):
        if ll == 'm':
            ii_end = ii-1
    #print(line2[:ii_end])
    r.close()

    CPUtime_tmp = float(line2[:ii_end])
    CPU_minu_list.append(CPUtime_tmp)
    for ii, vv in enumerate(v_list):
        if inpp[0] == vv:
            i_locate = ii
            break
    for jj, kk in enumerate(k_list):
        if inpp[1] == kk:
            j_locate = jj
            break
    ind_list.append([i_locate, j_locate])
    CPU_minu_mat[i_locate, j_locate] = CPUtime_tmp

countt = 0
for kk, cpuu in zip(k_list, CPU_minu_mat):
    di_CPUmin[str(kk)] = cpuu
df_CPUmin = pd.DataFrame(di_CPUmin)
df_CPUmin.set_index(keys=['velocity (m/s)'], inplace=True, drop=True)
print(df_CPUmin)

os.chdir(base_path)
# %%
# Pickle to Convergence

# %%
def pkl2vk(pkl_tmp):
    pkl_re= pkl_tmp.replace('pkl',' ')
    i_v_start = -1
    for ii, tt in enumerate(pkl_re):
        if tt == 'v':
            i_v_start = ii + 1
        elif tt == 'k':
            i_v_end =  ii -1
            i_k_start = ii +1
        elif tt == '.':
            i_k_end = ii
    if i_v_start < -0.001:
        return False, False
    v_re = float(pkl_tmp[i_v_start:i_v_end])
    k_re = float(pkl_tmp[i_k_start:i_k_end])
    return v_re, k_re

# %%

def pkl2conv(pkl_tmp):
    f = open(pkl_tmp, 'rb')
    y_res = pickle.load(f)
    y_sample = y_res[-10, 0]
    conv_bool = True
    if y_sample < 0.001:
        conv_bool = False
    f.close()
    return conv_bool

os.chdir(base_path)
os.chdir(f_path)
fnam_pkl_tmp = pkl_list[0]
pkl_conv_test = pkl2conv(fnam_pkl_tmp)
print('pkl2conv test:')
print(pkl_conv_test)

CONV_mat = np.zeros([len(v_list),len(k_list)])
di_CONV = {}
di_CONV['velocity (m/s)'] = v_list

CONV_list = []
ind_list_pkl = []

input_list_pkl = []
v_set = set([])
k_set = set([])
for pkk in pkl_list:
    v_tmp, k_tmp = pkl2vk(pkk)
    if type(v_tmp) == type(False):
        continue
    v_set.add(v_tmp)
    k_set.add(k_tmp)
    input_list_pkl.append([v_tmp, k_tmp])

for pkk,inpp in zip(pkl_list, input_list_pkl):
    conv_tmp = pkl2conv(pkk)

    CONV_list.append(conv_tmp)
    for ii, vv in enumerate(v_list):
        if inpp[0] == vv:
            i_locate = ii
            break
    for jj, kk in enumerate(k_list):
        if inpp[1] == kk:
            j_locate = jj
            break
    ind_list_pkl.append([i_locate, j_locate])
    CONV_mat[i_locate, j_locate] = conv_tmp


print(CONV_mat)

os.chdir(base_path)
# %%
CPU_av = np.sum(np.sum(CPU_minu_mat*CONV_mat))
Toto = np.sum(np.sum(CONV_mat))
print(CPU_av/Toto)
#print(Toto)
# %%
