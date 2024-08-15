import os
import numpy as np
from datetime import datetime
k_list = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
v_list = [0.001, 0.002, 0.005, 0.01]

a = datetime.now()
aa = a.strftime("%Y%m%d%H")


for kk in k_list:
    for vv in v_list:
        os.system('nohup python maonly_ori_argv.py '+ str(vv)+ ' ' + str(kk) +
                '> v' +str(vv)+'k' +str(kk) + '.log &') 
