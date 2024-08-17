import os
import numpy as np
from datetime import datetime
from move2dir import move
k_list = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
v_list = [0.001, 0.002, 0.005, 0.01]

a = datetime.now()
aa = a.strftime("%Y%m%d%H")

for kk in k_list:
    for vv in v_list:
        os.system('nohup python maonly_ori_argv.py '+ str(vv)+ ' ' + str(kk) +
                '> v' +str(vv)+'k' +str(kk) + '.log &') 

move('res_oriIAST')
