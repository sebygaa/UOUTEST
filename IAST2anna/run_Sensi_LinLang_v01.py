import os
import numpy as np
from datetime import datetime
import time
from move2dir import move
k_list = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
v_list = [0.001, 0.002, 0.005, 0.01]

a = datetime.now()
aa = a.strftime("%Y%m%d%H")

#base_path = os.getcwd()
#dirnam = 'res_IAST_or_'+ aa
#os.mkdir(dirnam)
#os.chdir(dirnam)
cc = 0
for kk in k_list:
    for vv in v_list:
        os.system('python maonly_LinInter_argv.py '+ str(vv)+ ' ' + str(kk) +
                '> v' +str(vv)+'k' +str(kk) + '.log') 