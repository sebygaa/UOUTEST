import os
import numpy as np

k_list = [ 0.01, 0.02, 0.05, 0.10, 0.20,]
v_list = [ 0.01, 0.02, 0.05, 0.10, 0.20,]

for kk in k_list:
    for vv in v_list:
        os.system('nohup python maonly_ori_argv.py '+ str(vv)+ ' ' + str(kk) +
                '> v' +str(vv)+'k' +str(kk) + '.log &') 

