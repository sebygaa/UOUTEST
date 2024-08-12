import os
import numy as np

k_list = [ 0.01, 0.02, 0.05, 0.10, 0.20,]
v_list = [ 0.01, 0.02, 0.05, 0.10, 0.20,]

for kk in k_list:
    for vv in v_list:
        os.system('python maonly_ori_arv.py '+ str[vv]+ ' ' + str[kk]

